import random
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
            **kwargs
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
                             x.is_file() and x.suffix != '.txt']
            self.instance_entries.extend(inst_img_path)

            if with_prior_preservation:
                class_img_path = [prompt_resolver(x, concept["class_prompt"], "class") for x in
                                  Path(concept["class_data_dir"]).iterdir() if
                                  x.is_file() and x.suffix != '.txt']
                self.class_entries.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_entries)
        self.num_instance_images = len(self.instance_entries)
        self.num_class_images = len(self.class_entries)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
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
