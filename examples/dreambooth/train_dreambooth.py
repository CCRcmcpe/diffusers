import atexit
import copy
import hashlib
import itertools
import logging
import math
import os
import random
import shutil
import string
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from modules.args import parser
from modules.datasets import DreamBoothDataset, PromptDataset, LatentsDataset

torch.backends.cudnn.benchmark = True

logging.basicConfig(level="INFO")
logger = get_logger("DB")


def get_params():
    args = parser.parse_args()
    config = None

    if args.resume:
        state_dir = Path(args.pretrained_model_name_or_path, "state")
        if not (state_dir / "state.pt").is_file() and (state_dir / "args.yaml").is_file():
            logger.warning("Checkpoint's state is broken, not resuming")
            args.resume = False

        logger.info("Trying to resume training, loading config from checkpoint")
        config_yaml = state_dir / "config.yaml"
        if config_yaml.is_file():
            if args.config is not None:
                logger.warning("Overriding checkpoint's config")
            else:
                config = OmegaConf.load(config_yaml)

    if config is None:
        config = OmegaConf.load(args.config)

    # config = DotMap(config)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args, config


def generate_class_images(concepts, args, noise_scheduler, accelerator):
    pipeline = None
    for concept in concepts:
        autogen_config = concept.class_set.auto_generate

        if not autogen_config.enabled:
            continue

        class_images_dir = Path(concept.class_set.path)
        class_images_dir.mkdir(parents=True, exist_ok=True)

        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images >= autogen_config.num_target:
            break

        num_new_images = autogen_config.num_target - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")

        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        if pipeline is None:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=AutoencoderKL.from_pretrained(
                    args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                    subfolder=None if args.pretrained_vae_name_or_path else "vae"
                ),
                scheduler=noise_scheduler,
                torch_dtype=torch_dtype,
                safety_checker=None
            )
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(accelerator.device)

        sample_dataset = PromptDataset([concept.class_set.prompt, autogen_config.negative_prompt], num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=autogen_config.batch_size)

        sample_dataloader = accelerator.prepare(sample_dataloader)

        with torch.autocast("cuda"), \
                torch.inference_mode(), \
                tqdm(total=num_new_images,
                     desc="Generating class images",
                     disable=not accelerator.is_local_main_process) as progress:
            for example in sample_dataloader:
                images = pipeline(prompt=example["prompt"][0][0],
                                  negative_prompt=example["prompt"][1][0],
                                  guidance_scale=autogen_config.cfg_scale,
                                  num_inference_steps=autogen_config.steps,
                                  num_images_per_prompt=len(example["prompt"][0])).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
                    progress.update()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_run_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=7))


def get_class(name: str):
    import importlib
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def get_optimizer(paramters_to_optimize, config, accelerator):
    params = dict(config.optimizer.params)

    lr_scale_config = config.optimizer.lr_scale
    if lr_scale_config.enabled:
        if lr_scale_config.method == "linear":
            params["lr"] *= config.gradient_accumulation_steps * config.batch_size * accelerator.num_processes
        elif lr_scale_config.method == "sqrt":
            params["lr"] *= math.sqrt(
                config.gradient_accumulation_steps * config.batch_size * accelerator.num_processes)
        else:
            raise ValueError()

    optimizer_class = get_class(config.optimizer.name)

    if "beta1" in params and "beta2" in params:
        params["betas"] = (params["beta1"], params["beta2"])
        del params["beta1"]
        del params["beta2"]

    optimizer = optimizer_class(paramters_to_optimize, **params)

    return optimizer


def get_lr_scheduler(config, optimizer) -> Any:
    lr_sched_config = config.optimizer.lr_scheduler
    scheduler = get_class(lr_sched_config.name)(optimizer, **lr_sched_config.params)

    if lr_sched_config.warmup.enabled:
        from modules.warmup_lr import WarmupLR
        scheduler = WarmupLR(scheduler,
                             init_lr=lr_sched_config.warmup.init_lr,
                             num_warmup=lr_sched_config.warmup.steps,
                             warmup_strategy=lr_sched_config.warmup.strategy)

    return scheduler


def main(args, config):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    run_id = generate_run_id() if args.run_id is None else args.run_id

    run_output_dir = output_dir / run_id
    run_output_dir.mkdir()

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=list(config.monitoring.monitors),
        logging_dir=run_output_dir / "logs",
    )

    logger.info(f"ID of current run: {run_id}")

    concepts = config.data.concepts
    not_used_read_txt = all(map(
        lambda c: not (c.instance_set.combine_prompt_from_txt or
                       c.class_set.combine_prompt_from_txt), concepts))

    if config.prior_preservation.enabled:
        if not_used_read_txt:
            logger.info("Running: DreamBooth (original paper method)")
        else:
            logger.info("Running: DreamBooth (alternative method)")
    elif not_used_read_txt:
        logger.info("Running: Equivalent to standard finetuning")
    else:
        logger.info("Running: [?]")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if config.train_text_encoder and config.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if config.seed:
        set_seed(config.seed)

    noise_scheduler = DDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    if config.prior_preservation.enabled:
        generate_class_images(concepts, args, noise_scheduler, accelerator)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer"
        )
    else:
        raise ValueError()

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )

    def encode_tokens(tokens):
        if config.clip_skip > 1:
            result = text_encoder(tokens, output_hidden_states=True, return_dict=True)
            result = result.hidden_states[-config.clip_skip]
            result = text_encoder.text_model.final_layer_norm(result)
        else:
            result = text_encoder(tokens)[0]

        return result

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    )

    vae.requires_grad_(False)
    if not config.train_text_encoder:
        text_encoder.requires_grad_(False)

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    params_to_optimize = (
        itertools.chain(unet.parameters(),
                        text_encoder.parameters()) if config.train_text_encoder else unet.parameters()
    )
    optimizer = get_optimizer(params_to_optimize, config, accelerator)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    base_step = 0
    base_epoch = 0

    if args.resume:
        checkpoint = torch.load(os.path.join(args.pretrained_model_name_or_path, "state", "state.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        base_step = checkpoint["total_steps"]
        base_epoch = checkpoint["total_epoch"]

    def get_dataset_class():
        if config.aspect_ratio_bucket.enabled:
            from modules.arb import DreamBoothDatasetWithARB
            return DreamBoothDatasetWithARB
        return DreamBoothDataset

    train_dataset = get_dataset_class()(
        concepts=concepts,
        tokenizer=tokenizer,
        with_prior_preservation=config.prior_preservation.enabled,
        size=config.data.resolution,
        center_crop=config.data.center_crop,
        pad_tokens=config.pad_tokens,
        bsz=config.batch_size,
        debug=config.aspect_ratio_bucket.debug,
        seed=config.seed,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if config.prior_preservation.enabled:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    cache_latents = config.cache_latents
    if config.aspect_ratio_bucket.enabled:
        cache_latents = False
        logger.warning("Latents cache disabled.")

        def collate_fn_wrap(examples):
            # workround for variable list
            if len(examples) == 1:
                examples = examples[0]
            return collate_fn(examples)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, collate_fn=collate_fn_wrap, pin_memory=True
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
        )

    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=torch.float32)
    if not config.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True,
                                                                 dtype=torch.float32)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if config.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(encode_tokens(batch["input_ids"]))
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x,
                                                       shuffle=True)

        del vae
        if not config.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    use_epochs_as_criteria = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if args.train_n_steps is None:
        args.train_n_steps = args.train_to_epochs * num_update_steps_per_epoch
        use_epochs_as_criteria = True

    if config.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if use_epochs_as_criteria:
        args.train_n_steps = args.train_to_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train_to_epochs = math.ceil(args.train_n_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    wandb_enabled = False
    if accelerator.is_main_process:
        if "wandb" in config.monitoring.monitors:
            import wandb
            wandb_enabled = True
        accelerator.init_trackers(args.project, config=dict(config))

    # Train!
    total_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Info of This Run *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Train to epochs = {args.train_to_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train_n_steps}")

    @atexit.register
    def on_exit():
        if config.saving.min_steps < local_steps < args.train_n_steps:
            print("Saving model...")
            on_step_end(from_interrupt=True)

    # Only show the progress bar once on each machine.
    if use_epochs_as_criteria:
        main_progress = tqdm(total=args.train_to_epochs, unit="epoch", disable=not accelerator.is_local_main_process)
        main_progress.set_description("Epochs")
    else:
        main_progress = tqdm(total=args.train_n_steps, unit="step", disable=not accelerator.is_local_main_process)
        main_progress.set_description("Steps")

    local_steps = global_steps = global_epochs = 0
    text_enc_context = nullcontext() if config.train_text_encoder else torch.no_grad()
    epoch_saved = False

    def on_step_end(from_interrupt=False):
        nonlocal epoch_saved

        save_checkpoint = (from_interrupt or
                           local_steps > config.saving.min_steps and
                           local_steps >= args.train_n_steps or
                           "interval_steps" in config.saving and global_steps % config.saving.interval_steps == 0 or
                           global_epochs > 0 and not epoch_saved and "interval_steps" not in config.saving and
                           global_epochs % config.saving.interval_epochs == 0)

        save_sample = (any(config.sampling.concepts) and
                       save_checkpoint or
                       global_steps % config.sampling.interval_steps == 0)

        if not (accelerator.is_main_process and save_sample):
            return

        # Create the pipeline using using the trained modules and save it.

        if config.train_text_encoder:
            text_enc_model = accelerator.unwrap_model(text_encoder)
        else:
            text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

        unet_unwrapped = accelerator.unwrap_model(unet)

        if config.saving.unet_half:
            unet_unwrapped = copy.deepcopy(unet_unwrapped).half()

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet_unwrapped,
            text_encoder=text_enc_model,
            vae=AutoencoderKL.from_pretrained(
                args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                subfolder=None if args.pretrained_vae_name_or_path else "vae"
            ),
            safety_checker=None,
            scheduler=noise_scheduler,
            torch_dtype=torch.float16
        )

        if save_checkpoint:
            save_dir = run_output_dir / str(global_steps)
            save_dir.mkdir()
            pipeline.save_pretrained(save_dir)

            state_dir = save_dir / "state"
            state_dir.mkdir()

            OmegaConf.save(config, state_dir / "config.yaml")

            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                "global_steps": global_steps,
                'global_epoch': global_epochs,
            }, state_dir / "state.pt")

            epoch_saved = True

            logger.info(f"[*] Checkpoint saved at {save_dir}")

            wandb_config = config.monitoring.wandb
            if accelerator.is_main_process and wandb_enabled and wandb_config.artifact:
                wandb_run = accelerator.get_tracker("wandb")

                model_artifact = wandb.Artifact('run_' + wandb_run.id + '_model', type='model', metadata={
                    'epochs_trained': global_epochs + 1,
                    'steps_trained': global_steps,
                    'project': args.project
                })
                model_artifact.add_dir(str(save_dir))
                wandb_run.log_artifact(model_artifact,
                                       aliases=['latest', 'last', f'epoch {global_epochs + 1}'])

                if wandb_config.remove_ckpt_after_upload:
                    shutil.rmtree(save_dir)

        if save_sample:
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            sample_dir = run_output_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            samples = []
            with torch.autocast("cuda"), torch.inference_mode():
                for concept in tqdm(config.sampling.concepts, unit="concept"):
                    g_cuda = torch.Generator(device=accelerator.device).manual_seed(concept.seed)
                    concept_samples = []
                    with tqdm(total=concept.num_samples + (concept.num_samples % config.sampling.batch_size),
                              desc=f"Generating samples") as progress:

                        for _ in range(math.ceil(concept.num_samples / config.sampling.batch_size)):
                            concept_samples.extend(pipeline(
                                prompt=concept.prompt,
                                negative_prompt=concept.negative_prompt,
                                guidance_scale=concept.cfg_scale,
                                num_inference_steps=concept.steps,
                                num_images_per_prompt=config.sampling.batch_size,
                                generator=g_cuda).images)
                            progress.update(config.sampling.batch_size)
                    samples.append((concept.prompt, concept_samples))
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for i, (_, images) in enumerate(samples):
                for j, image in enumerate(images):
                    image.save(sample_dir / f"{global_steps}_{i}_{j}.png")

            if wandb_enabled and config.monitoring.wandb.sample and any(samples):
                log_samples = {"samples": {prompt: [wandb.Image(x) for x in images] for prompt, images in samples}}
                accelerator.log(log_samples, global_steps, {"commit": False})

    for epoch in range(args.train_to_epochs):
        epoch_saved = False

        unet.train()
        if config.train_text_encoder:
            text_encoder.train()

        global_epochs = base_epoch + epoch

        sub_progress = tqdm(train_dataloader, unit="batch",
                            disable=not accelerator.is_local_main_process or not use_epochs_as_criteria)
        sub_progress.set_description(f"Epoch {global_epochs + 1}")

        for i, batch in enumerate(sub_progress):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if cache_latents:
                        latent_dist = batch[0][0]
                    else:
                        latent_dist = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist
                    latents = latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with text_enc_context:
                    if cache_latents:
                        if config.train_text_encoder:
                            encoder_hidden_states = encode_tokens(batch[0][1])
                        else:
                            encoder_hidden_states = batch[0][1]
                    else:
                        encoder_hidden_states = encode_tokens(batch["input_ids"])

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if config.prior_preservation.enabled:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + config.prior_preservation.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if config.gradient_clipping.enabled and accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if config.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, config.gradient_clipping.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            local_steps += 1
            lr_scheduler.step(epoch + i / len(train_dataloader))

            global_steps = base_step + local_steps

            logs = {"epoch": global_epochs + 1, "loss": loss.detach_().item(), "lr": lr_scheduler.get_lr()[0]}

            if use_epochs_as_criteria:
                l = logs.copy()
                del l["epoch"]
                l["total_steps"] = local_steps
                sub_progress.set_postfix(**l)
            else:
                main_progress.update()
                main_progress.set_postfix(**logs)

            on_step_end()

            accelerator.log(logs, step=global_steps)

            if local_steps >= args.train_n_steps:
                break

        accelerator.wait_for_everyone()

        if use_epochs_as_criteria:
            main_progress.update()

    accelerator.end_training()


if __name__ == "__main__":
    main(*get_params())
