For upstream info, refer to: https://github.com/ShivamShrirao/diffusers/blob/main/README.md

Mainly improved the dreambooth training script (see examples/dreambooth). But now it's pretty much a Stable Diffusion finetuner with dreambooth functionality.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gboJw5D3Qol_Fcm-XFd7CvUo99yM8U9y)

### Features

* Can run with 12G or less VRAM without losing speed with prior preservation loss enabled
* Additionally can use deepspeed to further reduce VRAM usage
* [Aspect Ratio Bucketing](https://github.com/NovelAI/novelai-aspect-ratio-bucketing)
* Support CLIP skip
* Support wandb logging
* Support per-image labels (both instance and class set)
* Deepdanbooru labeling script
* Cosine annealing LR scheduler
* You can use it without the dreambooth part (equivalent to standard finetuning process)
