import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfFolder, whoami
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--class_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt to specify images in the same class as provided instance images.",
    )

    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=11,
        help="CFG for save sample and class images generation.",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=28,
        help="The number of inference steps for save sample and class images generation.",
    )
    parser.add_argument(
        "--infer_batch_size", type=int, default=4,
        help="Batch size (per device) for save sample and class images generation."
    )

    parser.add_argument(
        "--pad_tokens",
        default=False,
        action="store_true",
        help="Flag to pad tokens to length 77.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for (not so) reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adamw_8bit", "adamw_ds", "sgdm", "sgdm_8bit"],
        help=(
            "The optimizer to use. _8bit optimizers require bitsandbytes, _ds optimizers require deepspeed."
        )
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--sgd_momentum", type=float, default=0.9, help="Momentum value for the SGDM optimizer")
    parser.add_argument("--sgd_dampening", type=float, default=0, help="Dampening value for the SGDM optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_cycles",
        type=int,
        default=None,
        help='The number of restarts to use. Default is no restarts. Only works with "cosine" and "cosine_with_restarts" lr scheduler.'
    )
    parser.add_argument(
        "--last_epoch",
        type=int,
        default=-1,
        help='The index of the last epoch for resuming training. Only works with "cosine" and "cosine_with_restarts" lr scheduler.'
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--save_interval", type=int, default=10_000, help="Save weights every N steps.")
    parser.add_argument("--save_min_steps", type=int, default=0, help="Start saving weights after N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--not_cache_latents", action="store_true",
                        help="Do not precompute and cache latents from VAE.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )

    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="Use wandb to watch training process.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="SD-Dreambooth-HF",
        help="Project name in your wandb.",
    )
    parser.add_argument(
        "--wandb_artifact",
        default=False,
        action="store_true",
        help="Upload saved weights to wandb.",
    )
    parser.add_argument(
        "--rm_after_wandb_saved",
        default=False,
        action="store_true",
        help="Remove saved weights from local machine after uploaded to wandb. Useful in colab.",
    )

    parser.add_argument(
        "--save_unet_half",
        default=False,
        action="store_true",
        help="Use half precision to save unet weights, saves storage.",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=1,
        help="Stop At last [n] layers of CLIP model when training."
    )

    parser.add_argument(
        "--read_prompt_from_txt",
        type=str,
        default=None,
        choices=["instance", "class", "both"],
        help="Merge with extra prompt from txt."
    )
    parser.add_argument(
        "--instance_insert_pos_regex",
        type=str,
        default=None,
        help="The regex used to match instance prompt in txt, so instance_prompt will be inserted at the match index"
    )
    parser.add_argument(
        "--class_insert_pos_regex",
        type=str,
        default=None,
        help="The regex used to match class prompt in txt, so class_prompt will be inserted at the match index"
    )

    parser.add_argument(
        "--use_aspect_ratio_bucket",
        default=False,
        action="store_true",
        help="Use aspect ratio bucketing as image processing strategy, which may improve the quality of outputs. Use it with --not_cache_latents"
    )
    parser.add_argument(
        "--debug_arb",
        default=False,
        action="store_true",
        help="Enable debug logging on aspect ratio bucket."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class BucketManager:
    def __init__(self, id_size_map, max_size=(768, 512), divisible=64, step_size=8, min_dim=256, base_res=(512, 512),
                 bsz=1, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=1024, debug=False):
        self.res_map = id_size_map
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0] / self.f) * (max_size[1] / self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = self.get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2 ** 32 - 1)
        self.epoch_prng = self.get_prng(epoch_seed)  # separate prng for sharding use for increased thread resilience
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()
        self.assign_buckets()
        self.start_epoch()

    @staticmethod
    def get_prng(seed):
        return np.random.RandomState(seed)

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        while (w / self.f) * (self.min_dim / self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w / self.f) * ((h + self.div) / self.f) <= self.max_tokens and (h + self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w) / float(h))
            w += self.div
        h = self.min_dim
        while (h / self.f) * (self.min_dim / self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h / self.f) * ((w + self.div) / self.f) <= self.max_tokens and (w + self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w) / float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def assign_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        self.buckets = {}
        self.aspect_errors = []
        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w) / float(h)
            bucket_id = np.abs(self.aspects - aspect).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                if self.debug:
                    self.aspect_errors.append(error)
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]
        if self.debug:
            timer = time.perf_counter() - timer
            self.aspect_errors = np.array(self.aspect_errors)
            print(f"skipped images: {skipped}")
            print(
                f"aspect error: mean {self.aspect_errors.mean()}, median {np.median(self.aspect_errors)}, max {self.aspect_errors.max()}")
            for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
                print(
                    f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, entries {len(self.buckets[bucket_id])}")
            print(f"assign_buckets: {timer:.5f}s")

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            timer = time.perf_counter()
        if world_size is not None:
            self.world_size = world_size
        if global_rank is not None:
            self.global_rank = global_rank

        # select ids for this epoch/rank
        index = np.array(sorted(list(self.res_map.keys())))
        index_len = index.shape[0]
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.bsz * self.world_size))]
        # print("perm", self.global_rank, index[0:16])
        index = index[self.global_rank::self.world_size]
        self.batch_total = index.shape[0] // self.bsz
        assert (index.shape[0] % self.bsz == 0)
        index = set(index)

        self.epoch = {}
        self.left_over = []
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = np.array([post_id for post_id in self.buckets[bucket_id] if post_id in index],
                                                 dtype=np.int64)
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    self.left_over.extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]
                if len(self.epoch[bucket_id]) == 0:
                    del self.epoch[bucket_id]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        # check if no data left or no epoch initialized
        if self.epoch is None or self.left_over is None or (
                len(self.left_over) == 0 and not bool(self.epoch)) or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            if len(self.left_over) >= self.bsz:
                bucket_probs = [len(self.left_over)] + [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
                bucket_ids = [-1] + bucket_ids
            else:
                bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_lens = bucket_probs
            bucket_probs = bucket_probs / bucket_probs.sum()
            bucket_ids = np.array(bucket_ids, dtype=np.int64)
            if bool(self.epoch):
                chosen_id = int(self.prng.choice(bucket_ids, 1, p=bucket_probs)[0])
            else:
                chosen_id = -1

            if chosen_id == -1:
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                self.prng.shuffle(self.left_over)
                batch_data = self.left_over[:self.bsz]
                self.left_over = self.left_over[self.bsz:]
                found_batch = True
            else:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    # return bucket batch and resolution
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.left_over.extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]

            assert (found_batch or len(self.left_over) >= self.bsz or bool(self.epoch))

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"bucket probs: " + ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs * 100))))
            print(f"chosen id: {chosen_id}")
            print(f"batch data: {batch_data}")
            print(f"resolution: {resolution}")
            print(f"get_batch: {timer:.5f}s")

        self.batch_delivered += 1
        return (batch_data, resolution)

    def generator(self):
        if self.batch_delivered >= self.batch_total:
            self.start_epoch()
        while self.batch_delivered < self.batch_total:
            yield self.get_batch()


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            tokenizer,
            with_prior_preservation=True,
            size=512,
            center_crop=False,
            num_class_images=None,
            pad_tokens=False,
            read_prompt_from_txt=None,
            instance_insert_pos_regex=None,
            class_insert_pos_regex=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.pad_tokens = pad_tokens

        self.instance_entries = []
        self.class_entries = []

        def combine_prompt(default_prompt, txt_prompt, prompt_type):
            match = None

            if prompt_type == "instance" and instance_insert_pos_regex is not None:
                match = re.search(instance_insert_pos_regex, txt_prompt)
            elif prompt_type == "class" and class_insert_pos_regex is not None:
                match = re.search(class_insert_pos_regex, txt_prompt)

            if match is None:
                return default_prompt + " " + txt_prompt

            idx = match.span()[0]
            return txt_prompt[:idx] + default_prompt + (" " + txt_prompt[idx:]) if len(txt_prompt[idx:]) > 0 else ""

        def prompt_resolver(x, default, prompt_type):
            entry = (x, default)

            if read_prompt_from_txt is None or read_prompt_from_txt != prompt_type and read_prompt_from_txt != "both":
                return entry

            content = Path(x).with_suffix('.txt').read_text()
            combined_prompt = combine_prompt(default, content, prompt_type)

            entry = (x, combined_prompt)

            return entry

        for concept in concepts_list:
            inst_img_path = [prompt_resolver(x, concept["instance_prompt"], "instance") for x in
                             Path(concept["instance_data_dir"]).iterdir() if
                             x.is_file()]
            self.instance_entries.extend(inst_img_path)

            if with_prior_preservation:
                class_img_path = [prompt_resolver(x, concept["class_prompt"], "class") for x in
                                  Path(concept["class_data_dir"]).iterdir() if
                                  x.is_file()]
                self.class_entries.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_entries)
        self.num_instance_images = len(self.instance_entries)
        self.num_class_images = len(self.class_entries)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BOX),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @staticmethod
    def read_img(filepath) -> Image:
        img = Image.open(filepath)

        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img

    def tokenize(self, prompt):
        return self.tokenizer(
            prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.instance_entries[index % self.num_instance_images]
        instance_image = self.read_img(instance_path)
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenize(instance_prompt)

        if self.with_prior_preservation:
            class_path, class_prompt = self.class_entries[index % self.num_class_images]
            class_image = self.read_img(class_path)
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenize(class_prompt)

        return example


class DreamBoothDatasetWithARB(torch.utils.data.IterableDataset, DreamBoothDataset):
    def __init__(self, bsz=1, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug

        self.prompt_cache = {}

        instance_id_size_map = self.get_path_size_map(str(item[0]) for item in self.instance_entries)
        self.instance_bucket_manager = BucketManager(instance_id_size_map, bsz=bsz, debug=debug).generator()

        if self.with_prior_preservation:
            self.class_id_bucket_map = {}
            class_id_size_map = self.get_path_size_map(str(item[0]) for item in self.class_entries)
            for batch, size in BucketManager(class_id_size_map, bsz=1, debug=debug).generator():
                self.class_id_bucket_map.setdefault(size, []).extend([batch])

        # cache prompts for reading
        for path, prompt in self.instance_entries + self.class_entries:
            self.prompt_cache[path] = prompt

    @staticmethod
    def get_path_size_map(images_paths):
        path_size_map = {}

        for image_path in tqdm(images_paths, desc="Loading resolution from images"):
            with Image.open(image_path) as img:
                size = img.size
            path_size_map[image_path] = size

        return path_size_map

    def transform(self, img, size, center_crop=False):
        x, y = img.size
        short, long = (x, y) if x <= y else (y, x)

        w, h = size
        min_crop, max_crop = (w, h) if w <= h else (h, w)
        ratio_src, ratio_dst = int(long / short), int(max_crop / min_crop)

        if ratio_src > ratio_dst:
            new_w, new_h = (min_crop, int(min_crop * ratio_src)) if x < y else (int(min_crop * ratio_src), min_crop)
        elif ratio_src < ratio_dst:
            new_w, new_h = (max_crop, int(max_crop / ratio_src)) if x > y else (int(max_crop / ratio_src), max_crop)
        else:
            new_w, new_h = w, h

        image_transforms = transforms.Compose([
            transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BOX),
            transforms.CenterCrop((h, w)) if center_crop else transforms.RandomCrop((h, w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        new_img = image_transforms(img)

        if self.debug:
            print(x, y, w, h, "->", new_img.shape)
            import uuid, torchvision
            filename = str(uuid.uuid4())
            torchvision.utils.save_image(new_img, f"/tmp/{filename}_1.jpg")
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(img), f"/tmp/{filename}_2.jpg")
            print(f"saved: /tmp/{filename}")

        return new_img

    def __iter__(self):
        for batch, size in self.instance_bucket_manager:
            result = []

            for instance_path in batch:
                example = {}
                instance_prompt = self.prompt_cache[instance_path]
                instance_image = self.read_img(instance_path)
                example["instance_images"] = self.transform(instance_image, size)
                example["instance_prompt_ids"] = self.tokenize(instance_prompt)

                if self.with_prior_preservation:
                    if not any(self.class_id_bucket_map[size]):
                        print(f"Warning: no class image with {size} exists. Will use instance image as is.")
                        example["class_images"] = self.transform(instance_image, size)
                        example["class_prompt_ids"] = self.tokenize(instance_prompt)
                        result.append(example)
                        continue

                    class_path = random.choice(self.class_id_bucket_map[size])
                    class_prompt = self.prompt_cache[class_path]
                    class_image = self.read_img(class_path)
                    example["class_images"] = self.transform(class_image, size)
                    example["class_prompt_ids"] = self.tokenize(class_prompt)

                result.append(example)
            yield result


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


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


def generate_class_images(args, accelerator):
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

        with torch.autocast("cuda"), torch.inference_mode():
            for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(prompt=example["prompt"][0][0],
                                  negative_prompt=example["prompt"][1][0],
                                  guidance_scale=args.guidance_scale,
                                  num_inference_steps=args.infer_steps,
                                  num_images_per_prompt=len(example["prompt"][0])).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    logging_dir = Path(args.output_dir, "0", args.logging_dir)

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
        logging_dir=logging_dir,
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

    if args.with_prior_preservation:
        generate_class_images(args, accelerator)

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

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

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

    noise_scheduler = DDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    dataset_class = DreamBoothDatasetWithARB if args.use_aspect_ratio_bucket else DreamBoothDataset
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
        print("Latents cache disabled.")

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

    def save_weights(step, epoch):
        # Create the pipeline using using the trained modules and save it.
        if not accelerator.is_main_process:
            return

        if args.train_text_encoder:
            text_enc_model = accelerator.unwrap_model(text_encoder)
        else:
            text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)

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
            scheduler=scheduler,
            torch_dtype=torch.float16
        )
        save_dir = os.path.join(args.output_dir, f"{step}")
        pipeline.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

        if args.save_sample_prompt is not None:
            pipeline = pipeline.to(accelerator.device)
            g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            pipeline.set_progress_bar_config(disable=True)
            sample_dir = os.path.join(save_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            with torch.autocast("cuda"), torch.inference_mode():
                for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                    images = pipeline(
                        prompt=args.save_sample_prompt,
                        negative_prompt=args.save_sample_negative_prompt,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.infer_steps,
                        num_images_per_prompt=args.infer_batch_size,
                        generator=g_cuda).images
                    for k, image in enumerate(images):
                        image.save(os.path.join(sample_dir, f"{i * args.infer_batch_size + k}.png"))
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[*] Weights saved at {save_dir}")

        if args.wandb:
            accelerator.log({"samples": [wandb.Image(x) for x in images]}, step=global_step)

            if args.wandb_artifact:
                model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
                    'epochs_trained': epoch + 1,
                    'project': run.project
                })
                model_artifact.add_dir(save_dir)
                wandb.log_artifact(model_artifact,
                                   aliases=['latest', 'last', f'epoch {epoch + 1}'])

                if args.rm_after_wandb_saved:
                    shutil.rmtree(save_dir)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
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
                loss_avg.update(loss.detach_(), bsz)

            logs = {"epoch": epoch + 1, "loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            step_saved = False

            if global_step > 0 and not global_step % args.save_interval:
                save_weights(global_step, epoch)
                step_saved = True

            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps and not step_saved:
                save_weights(global_step, epoch)
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
