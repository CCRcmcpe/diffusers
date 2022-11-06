import atexit
import copy
import hashlib
import itertools
import json
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
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from modules.args import parser
from modules.datasets import DreamBoothDataset, PromptDataset, LatentsDataset

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)


def parse_args(input_argv=None):
    args = parser.parse_args(input_argv)

    if args.resume:
        state_dir = Path(args.pretrained_model_name_or_path, "state")
        if not (state_dir / "state.pt").is_file() and (state_dir / "args.yaml").is_file():
            logger.warning("Checkpoint's state is broken, not resuming")
            args.resume = False

        logger.info("Trying to resume training, loading config from checkpoint")
        config = state_dir / "args.yaml"
        if config.is_file():
            if args.config is not None:
                logger.warning("Overriding checkpoint's config")
            else:
                args.config = str(config)

    if args.config is not None and Path(args.config).is_file():
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader)
        parser.set_defaults(**config)
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_optimizer_class(optimizer_name: str) -> Any:
    def try_import_bnb():
        try:
            import bitsandbytes as bnb
            return bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit optimizers, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    def try_import_ds():
        try:
            import deepspeed
            return deepspeed
        except ImportError:
            raise ImportError(
                "Failed to import Deepspeed"
            )

    name = optimizer_name.lower()

    if name == "adamw":
        return torch.optim.AdamW
    elif name == "adamw_8bit":
        return try_import_bnb().optim.AdamW8bit
    elif name == "adamw_ds":
        return try_import_ds().ops.adam.DeepSpeedCPUAdam
    elif name == "sgdm":
        return torch.optim.sgd
    elif name == "sgdm_8bit":
        return try_import_bnb().optim.SGD8bit
    else:
        raise ValueError("WTF is that optimizer")


def generate_class_images(args, noise_scheduler, accelerator):
    pipeline = None
    for concept in args.concepts_list:
        class_images_dir = Path(concept["class_data_dir"])
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images >= args.num_class_images:
            break

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

        num_new_images = args.num_class_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset([concept["class_prompt"], concept["class_negative_prompt"]], num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.infer_batch_size)

        sample_dataloader = accelerator.prepare(sample_dataloader)

        with torch.autocast("cuda"), \
                torch.inference_mode(), \
                tqdm(total=num_new_images,
                     desc="Generating class images",
                     disable=not accelerator.is_local_main_process) as progress:
            for example in sample_dataloader:
                images = pipeline(prompt=example["prompt"][0][0],
                                  negative_prompt=example["prompt"][1][0],
                                  guidance_scale=args.guidance_scale,
                                  num_inference_steps=args.infer_steps,
                                  num_images_per_prompt=len(example["prompt"][0])).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
                    progress.update()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    run_id = args.run_id if args.run_id is not None else "".join(
        random.choices(string.ascii_uppercase + string.digits, k=7))
    run_output_dir = output_dir / run_id
    run_output_dir.mkdir()

    loggers = ["tensorboard"]

    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project)
        wandb.config = vars(args)
        loggers.append("wandb")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=loggers,
        logging_dir=run_output_dir / "logs",
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "class_negative_prompt": args.class_negative_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    noise_scheduler = DDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.with_prior_preservation:
        generate_class_images(args, noise_scheduler, accelerator)

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
        if args.clip_skip > 1:
            result = text_encoder(tokens, output_hidden_states=True, return_dict=True)
            result = result.hidden_states[-args.clip_skip]
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
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr_linear:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    elif args.scale_lr:
        args.learning_rate *= math.sqrt(
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    optimizer_class = get_optimizer_class(args.optimizer)

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )

    if "adam" in args.optimizer.lower():
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
    elif "sgd" in args.optimizer.lower():
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            dampening=args.sgd_dampening,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(args.optimizer)

    base_step = 0
    base_epoch = 0

    if args.resume:
        checkpoint = torch.load(os.path.join(args.pretrained_model_name_or_path, "state", "state.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        base_step = checkpoint["total_steps"]
        base_epoch = checkpoint["total_epoch"]

    dataset_class = __import__(
        "modules.arb.DreamBoothDatasetWithARB") if args.use_aspect_ratio_bucket else DreamBoothDataset
    train_dataset = dataset_class(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        pad_tokens=args.pad_tokens,
        read_prompt_from_txt=args.read_prompt_from_txt,
        instance_insert_pos_regex=args.instance_insert_pos_regex,
        class_insert_pos_regex=args.class_insert_pos_regex,
        bsz=args.train_batch_size,
        debug=args.debug_arb,
        seed=args.seed,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
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

    if args.use_aspect_ratio_bucket:
        args.not_cache_latents = True
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
            train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.not_cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True,
                                                                 dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(encode_tokens(batch["input_ids"]))
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x,
                                                       shuffle=True)

        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_cycles is None:
        if args.lr_scheduler.lower() == "cosine":
            args.lr_cycles = 0.5
        if args.lr_scheduler.lower() == "cosine_with_restarts":
            args.lr_cycles = 1

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_cycles,
        last_epoch=args.last_epoch
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    @atexit.register
    def on_exit():
        if args.save_min_steps < step < args.max_train_steps:
            print("Saving model...")
            on_step_end(from_interrupt=True)

    # Only show the progress bar once on each machine.
    if overrode_max_train_steps:
        main_progress = tqdm(total=args.num_train_epochs, unit="epoch", disable=not accelerator.is_local_main_process)
        main_progress.set_description("Epochs")
    else:
        main_progress = tqdm(total=args.max_train_steps, unit="step", disable=not accelerator.is_local_main_process)
        main_progress.set_description("Steps")

    step = global_step = global_epoch = 0
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    epoch_saved = False

    def on_step_end(from_interrupt=False):
        nonlocal epoch_saved

        save_checkpoint = (step > args.save_min_steps and
                           step >= args.max_train_steps or
                           (args.save_interval is not None and global_step % args.save_interval == 0 or
                            global_epoch > 0 and global_epoch % args.save_interval_epochs == 0 and not epoch_saved)) or from_interrupt

        save_sample = (args.save_sample_prompt is not None and
                       (save_checkpoint or args.sample_interval is not None and
                        global_step % args.sample_interval == 0))

        if not (accelerator.is_main_process and save_sample):
            return

        # Create the pipeline using using the trained modules and save it.

        if args.train_text_encoder:
            text_enc_model = accelerator.unwrap_model(text_encoder)
        else:
            text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

        unet_unwrapped = accelerator.unwrap_model(unet)

        if args.save_unet_half:
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
            save_dir = run_output_dir / str(global_step)
            save_dir.mkdir()
            pipeline.save_pretrained(save_dir)

            state_dir = save_dir / "state"
            state_dir.mkdir()

            with open(state_dir / "args.yaml", "w") as f:
                yaml.dump(args.__dict__, f, Dumper, indent=2)

            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                "global_steps": global_step,
                'global_epoch': global_epoch,
            }, state_dir / "state.pt")

            epoch_saved = True

            logger.info(f"[*] Checkpoint saved at {save_dir}")

            if args.wandb and args.wandb_artifact:
                model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
                    'epochs_trained': global_epoch + 1,
                    'steps_trained': global_step,
                    'project': run.project
                })
                model_artifact.add_dir(str(save_dir))
                wandb.log_artifact(model_artifact,
                                   aliases=['latest', 'last', f'epoch {global_epoch + 1}'])

                if args.rm_after_wandb_saved:
                    shutil.rmtree(save_dir)

        if save_sample:
            pipeline = pipeline.to(accelerator.device)
            g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            pipeline.set_progress_bar_config(disable=True)
            sample_dir = run_output_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            samples = []
            with torch.autocast("cuda"), \
                    torch.inference_mode(), \
                    tqdm(total=args.n_save_sample + (args.n_save_sample % args.infer_batch_size),
                         desc="Generating samples") as progress:
                for _ in range(math.ceil(args.n_save_sample / args.infer_batch_size)):
                    samples.extend(pipeline(
                        prompt=args.save_sample_prompt,
                        negative_prompt=args.save_sample_negative_prompt,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.infer_steps,
                        num_images_per_prompt=args.infer_batch_size,
                        generator=g_cuda).images)
                    progress.update(args.infer_batch_size)
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for i, image in enumerate(samples):
                image.save(sample_dir / f"{global_step}_{i}.png")

            if args.wandb and args.wandb_sample and any(samples):
                wandb.log({"samples": [wandb.Image(x) for x in samples]}, step=global_step, commit=False)

    for epoch in range(args.num_train_epochs):
        epoch_saved = False

        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        global_epoch = base_epoch + epoch

        sub_progress = tqdm(train_dataloader, unit="batch",
                            disable=not accelerator.is_local_main_process or not overrode_max_train_steps)
        sub_progress.set_description(f"Epoch {global_epoch + 1}")

        for batch in sub_progress:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if not args.not_cache_latents:
                        latent_dist = batch[0][0]
                    else:
                        latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
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
                    if not args.not_cache_latents:
                        if args.train_text_encoder:
                            encoder_hidden_states = encode_tokens(batch[0][1])
                        else:
                            encoder_hidden_states = batch[0][1]
                    else:
                        encoder_hidden_states = encode_tokens(batch["input_ids"])

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = (
                #         itertools.chain(unet.parameters(), text_encoder.parameters())
                #         if args.train_text_encoder
                #         else unet.parameters()
                #     )
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            global_step = base_step + step

            logs = {"epoch": global_epoch + 1, "loss": loss.detach_().item(), "lr": lr_scheduler.get_last_lr()[0]}

            if overrode_max_train_steps:
                l = logs.copy()
                del l["epoch"]
                l["total_steps"] = step
                sub_progress.set_postfix(**l)
            else:
                main_progress.update()
                main_progress.set_postfix(**logs)

            on_step_end()

            accelerator.log(logs, step=global_step)

            if step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        if overrode_max_train_steps:
            main_progress.update()

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
