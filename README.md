




































































































































  
    
    
    
    
    
    
    
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    "runwayml/stable-diffusion-inpainting",
    "runwayml/stable-diffusion-v1-5", 
    "runwayml/stable-diffusion-v1-5", revision="bf16", dtype=jax.numpy.bfloat16
    "runwayml/stable-diffusion-v1-5", revision="flax", dtype=jax.numpy.bfloat16
    </a>
    </a>
    </a>
    <a href="CODE_OF_CONDUCT.md">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE">
    <a href="https://github.com/huggingface/diffusers/releases">
    <br>
    <br>
    <br>
    <br>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
    <em> Figure from ImageGen (https://imagen.research.google/). </em>
    <em> Sampling and training algorithms. Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
    <img src="https://github.com/huggingface/diffusers/raw/main/docs/source/imgs/diffusers_library.jpg" width="400"/>
    <img src="https://user-images.githubusercontent.com/10695622/174348898-481bd7c2-5457-4830-89bc-f0907756f64c.jpeg" width="550"/>
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <img src="https://user-images.githubusercontent.com/10695622/174349706-53d58acc-a4d1-4cda-b3e8-432d9dc7ad38.png" width="800"/>
    model_id_or_path,
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")
    revision="fp16",
    revision="fp16", 
    revision="fp16", 
    torch_dtype=torch.float16,
    torch_dtype=torch.float16,
    torch_dtype=torch.float16,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  diffusion models on an image dataset, with explanatory graphics. 
  howpublished = {\url{https://github.com/huggingface/diffusers}}
  journal = {GitHub repository},
  publisher = {GitHub},
  Take a look at this notebook to learn how to use the pipeline abstraction, which takes care of everything (model, scheduler, noise handling) for you, and also to understand each independent building block in the library.
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim).
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)
- [Text-to-Image Latent Diffusion](https://huggingface.co/CompVis/ldm-text2im-large-256)
- [Unconditional Diffusion with continuous scheduler](https://huggingface.co/google/ncsnpp-ffhq-1024)
- [Unconditional Diffusion with discrete scheduler](https://huggingface.co/google/ddpm-celebahq-256)
- [Unconditional Latent Diffusion](https://huggingface.co/CompVis/ldm-celebahq-256)
- BDDMPipeline for spectrogram-to-sound vocoding
- Diffusers for audio
- Diffusers for molecule generation (initial work happening in https://github.com/huggingface/diffusers/pull/54)
- Diffusers for reinforcement learning (initial work happening in https://github.com/huggingface/diffusers/pull/105).
- Diffusers for video generation
- Diffusers is **modality independent** and focuses on providing pretrained models and tools to build systems that generate **continuous outputs**, *e.g.* vision and audio.
- Diffusion models and schedulers are provided as concise, elementary building blocks. In contrast, diffusion pipelines are a collection of end-to-end diffusion systems that can be used out-of-the-box, should stay as close as possible to their original implementation and can include components of another library, such as text-encoders. Examples for diffusion pipelines are [Glide](https://github.com/openai/glide-text2im) and [Latent Diffusion](https://github.com/CompVis/latent-diffusion).
- Dreambooth. Another technique to capture new concepts in Stable Diffusion. This method fine-tunes the UNet (and, optionally, also the text encoder) of the pipeline to achieve impressive results. Please, refer to [our training example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) and [training report](https://huggingface.co/blog/dreambooth) for additional details and training recommendations.
- Full Stable Diffusion fine-tuning. If you have a more sizable dataset with a specific look or style, you can fine-tune Stable Diffusion so that it outputs images following those examples. This was the approach taken to create [a Pokémon Stable Diffusion model](https://huggingface.co/justinpinkney/pokemon-stable-diffusion) (by Justing Pinkney / Lambda Labs), [a Japanese specific version of Stable Diffusion](https://huggingface.co/spaces/rinna/japanese-stable-diffusion) (by [Rinna Co.](https://github.com/rinnakk/japanese-stable-diffusion/) and others. You can start at [our text-to-image fine-tuning example](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) and go from there.
- GLIDEPipeline to support OpenAI's GLIDE model
- Grad-TTS for text to audio generation / conditional audio generation
- Multiple types of models, such as UNet, can be used as building blocks in an end-to-end diffusion system (see [src/diffusers/models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)).
- Readability and clarity is preferred over highly optimized code. A strong importance is put on providing readable, intuitive and elementary code design. *E.g.*, the provided [schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers) are separated from the provided [models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models) and provide well-commented code that can be read alongside the original paper.
- See [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for general opportunities to contribute
- See [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) to contribute exciting new diffusion models / diffusion pipelines
- See [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)
- State-of-the-art diffusion pipelines that can be run in inference with just a couple of lines of code (see [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)). Check [this overview](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/README.md#pipelines-summary) to see all supported pipelines and their corresponding official papers.
- Textual Inversion. Capture novel concepts from a small set of sample images, and associate them with new "words" in the embedding space of the text encoder. Please, refer to [our training examples](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) or [documentation](https://huggingface.co/docs/diffusers/training/text_inversion) to try for yourself.
- The [Getting started with Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) notebook, which showcases an end-to-end example of usage for diffusion models, schedulers and pipelines.
- The [Training a diffusers model](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook summarizes diffusion models training methods. This notebook takes a step-by-step approach to training your
- Training examples to show how to train the most popular diffusion model tasks (see [examples](https://github.com/huggingface/diffusers/tree/main/examples), *e.g.* [unconditional-image-generation](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation)).
- Various noise schedulers that can be used interchangeably for the preferred speed vs. quality trade-off in inference (see [src/diffusers/schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)).
# !pip install diffusers["torch"]
# !pip install diffusers["torch"] transformers
# and pass `model_id_or_path="./stable-diffusion-v1-5"`.
# disable the following line if you run on CPU
# let's download an initial image
# load model and scheduler
# load model and scheduler
# load the pipeline
# make sure you're logged in with `huggingface-cli login`
# or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
# run pipeline in inference (sample random noise and denoise)
# run pipeline in inference (sample random noise and denoise)
# save image
# save image
# shard inputs and rng
# shard inputs and rng
## Citation
## Contributing
## Credits
## Definitions
## Fine-Tuning Stable Diffusion
## In the works
## Installation
## Other Examples
## Philosophy
## Quickstart
## Stable Diffusion Community Pipelines
## Stable Diffusion is fully compatible with `diffusers`!  
### For Flax
### For PyTorch
### Image-to-Image text-guided generation with Stable Diffusion
### In-painting using Stable Diffusion
### JAX/Flax
### Running Code
### Text-to-Image generation with Stable Diffusion
### Tweak prompts reusing seeds and latents
### Web Demos
#### Running the model locally
(after having [accepted the license](https://huggingface.co/runwayml/stable-diffusion-v1-5)) and pass
)
)
)
)
)
* [image-to-image generation with Stable Diffusion](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg),
* [Model-based reinforcement learning](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/reinforcement_learning_with_diffusers.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg),
* [Molecule conformation generation](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/geodiff_molecule_conformation.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg),
* [tweak images via repeated Stable Diffusion seeds](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg),
**Apple Silicon (M1/M2) support**
**Diffusers for Other Modalities**:
**Diffusion Pipeline**: End-to-end pipeline that includes multiple diffusion models, possible text encoders, ...
**Models**: Neural network that models $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ (see image below) and is trained end-to-end to *denoise* a noisy input to an image.
**Note**:
**Other Image Notebooks**:
**Schedulers**: Algorithm class for both **inference** and **training**.
**With `conda`**
**With `pip`**
**With `pip`**
*Examples*: [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), [PNDM](https://arxiv.org/abs/2202.09778), [DEIS](https://arxiv.org/abs/2204.13902)
*Examples*: Glide, Latent-Diffusion, Imagen, DALL-E 2
*Examples*: UNet, Conditioned UNet, 3D UNet, Transformer UNet
@misc{von-platen-etal-2022-diffusers,
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```bash
```bash
```bash
```bash
```bibtex
```python
```python
```python
```python
```python
```python
```python
```python
```python
```python
```python
```sh
| Composable diffusion | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Shuang59/Composable-Diffusion)           	|
| Conditional generation from sketch  	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/huggingface/diffuse-the-rest)           	|
| DDPM with different schedulers 	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/fusing/celeba-diffusion)           	|
| Faces generator                	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CompVis/celeba-latent-diffusion)    	|
| Model                          	| Hugging Face Spaces                                                                                                                                               	|
| Text-to-Image Latent Diffusion 	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CompVis/text2img-latent-diffusion) 	|
|--------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
}
</p>
<p align="center">
<p align="center">
<p align="center">
<p align="center">
<p align="center">
<p>
<p>
<p>
<p>
A few pipeline components are already being worked on, namely:
Also, say 👋 in our public Discord channel <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>. We discuss the hottest trends about diffusion models, help each other with contributions, personal projects or
and have a look into the [release notes](https://github.com/huggingface/diffusers/releases/tag/v0.2.0).
as a modular toolbox for inference and training of diffusion models.
Assuming the folder is stored locally under `./stable-diffusion-v1-5`, you can also run stable diffusion
conda install -c conda-forge diffusers
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
ddpm.to(device)
def download_image(url):
device = "cuda"
device = "cuda"
device = "cuda"
Diffusers offers a JAX / Flax implementation of Stable Diffusion for very fast inference. JAX shines specially on TPU hardware because each TPU server has 8 accelerators working in parallel, but it runs great on GPUs too.
Fine-tuning techniques make it possible to adapt Stable Diffusion to your own dataset, or add new subjects to it. These are some of the techniques supported in `diffusers`:
First let's install
For more details, check out [the Stable Diffusion notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)
For the first release, 🤗 Diffusers focuses on text-to-image diffusion techniques. However, diffusers can be used for much more than that! Over the upcoming releases, we'll be focusing on:
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from diffusers import DiffusionPipeline
from diffusers import FlaxStableDiffusionPipeline
from diffusers import FlaxStableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline
from flax.jax_utils import replicate
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from flax.training.common_utils import shard
from io import BytesIO
from io import BytesIO
from PIL import Image
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
git lfs install
huggingface-cli login
If you are limited by GPU memory, you might want to consider chunking the attention computation in addition 
If you are limited by TPU memory, please make sure to load the `FlaxStableDiffusionPipeline` in `bfloat16` precision instead of the default `float32` precision as done above. You can do so by telling diffusers to load the weights from "bf16" branch.
If you don't want to login to Hugging Face, you can also simply download the model folder
If you just want to play around with some web demos, you can try out the following 🚀 Spaces:
If you want to contribute to this library, please check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
If you want to run Stable Diffusion on CPU or you want to have maximum precision on GPU, 
If you want to run the code yourself 💻, you can try out:
If you wish to use a different scheduler (e.g.: DDIM, LMS, PNDM/PLMS), you can instantiate
image = ddpm().images[0]
image = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images[0]
image = pipe(prompt).images[0]  
image = pipe(prompt).images[0]  
image = pipe(prompt).images[0]  
image = pipe(prompt).images[0]  
image = pipe(prompt).images[0]  
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
image.save("astronaut_rides_horse.png")
image.save("astronaut_rides_horse.png")
image.save("ddpm_generated_image.png")
image.save("squirrel.png")
images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images
images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
images[0].save("fantasy_landscape.png")
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
import jax
import jax
import numpy as np
import numpy as np
import PIL
import requests
import requests
import torch
import torch
import torch
In order to get started, we recommend taking a look at two notebooks:
init_image = download_image(img_url).resize((512, 512))
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
it before the pipeline and pass it to `from_pretrained`.
just hang out ☕.
ldm = DiffusionPipeline.from_pretrained(model_id)
ldm = ldm.to(device)
mask_image = download_image(mask_url).resize((512, 512))
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
model_id = "CompVis/ldm-text2im-large-256"
model_id = "google/ddpm-celebahq-256"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
More precisely, 🤗 Diffusers offers:
num_inference_steps = 50
num_inference_steps = 50
num_samples = jax.device_count()
num_samples = jax.device_count()
Our [Community Examples folder](https://github.com/huggingface/diffusers/tree/main/examples/community) contains many ideas worth exploring, like interpolating to create animated videos, using CLIP Guidance for additional prompt fidelity, term weighting, and much more! [Take a look](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview) and [contribute your own](https://huggingface.co/docs/diffusers/using-diffusers/contribute_pipeline).
params = replicate(params)
params = replicate(params)
pip install --upgrade diffusers transformers scipy
pip install --upgrade diffusers[flax]
pip install --upgrade diffusers[torch]
pipe = pipe.to("cuda")
pipe = pipe.to("cuda")
pipe = pipe.to("cuda")
pipe = pipe.to("cuda")
pipe = pipe.to("cuda")
pipe = pipe.to(device)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
pipe = StableDiffusionInpaintPipeline.from_pretrained(
pipe = StableDiffusionPipeline.from_pretrained(
pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
pipe.enable_attention_slicing()
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
please run the model in the default *full-precision* setting:
Please, refer to [the documentation](https://huggingface.co/docs/diffusers/optimization/mps).
Please, visit the [model card](https://huggingface.co/runwayml/stable-diffusion-inpainting), read the license carefully and tick the checkbox if you agree. Note that this is an additional license, you need to accept it even if you accepted the text-to-image Stable Diffusion license in the past. You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section](https://huggingface.co/docs/hub/security-tokens) of the documentation.
precision while being roughly twice as fast and requiring half the amount of GPU RAM.
prng_seed = jax.random.PRNGKey(0)
prng_seed = jax.random.PRNGKey(0)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt = "A fantasy landscape, trending on artstation"
prompt = "A painting of a squirrel eating a burger"
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
prompt = num_samples * [prompt]
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids = shard(prompt_ids)
prompt_ids = shard(prompt_ids)
response = requests.get(url)
Run this command to log in with your HF Hub token if you haven't before (you can skip this step if you prefer to run the model locally, follow [this](#running-the-model-locally) instead)
Running the pipeline with the default PNDMScheduler:
See the [model card](https://huggingface.co/CompVis/stable-diffusion) for more information.
Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), [LAION](https://laion.ai/) and [RunwayML](https://runwayml.com/). It's trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 4GB VRAM.
Textual Inversion is a technique for capturing novel concepts from a small number of example images in a way that can later be used to control text-to-image pipelines. It does so by learning new 'words' in the embedding space of the pipeline's text encoder. These special words can then be used within text prompts to achieve very fine-grained control of the resulting images. 
The `StableDiffusionImg2ImgPipeline` lets you pass a text prompt and an initial image to condition the generation of new images.
The `StableDiffusionInpaintPipeline` lets you edit specific parts of an image by providing a mask and a text prompt. It uses a model optimized for this particular task, whose license you need to accept before use.
The class provides functionality to compute previous image according to alpha, beta schedule as well as predict noise for training. Also known as **Samplers**.
The following snippet should result in less than 4GB VRAM.
the path to the local folder to the `StableDiffusionPipeline`.
The release of Stable Diffusion as an open source model has fostered a lot of interesting ideas and experimentation. 
There are many ways to try running Diffusers! Here we outline code-focused tools (primarily using `DiffusionPipeline`s and Google Colab) and interactive web-tools.
This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:
to using `fp16`.
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
We ❤️  contributions from the open-source community! 
We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.
We recommend using the model in [half-precision (`fp16`)](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) as it gives almost always the same results as full
We want diffusers to be a toolbox useful for diffusers models in general; if you find yourself limited in any way by the current API, or would like to see additional models, schedulers, or techniques, please open a [GitHub issue](https://github.com/huggingface/diffusers/issues) mentioning what you would like to see.
without requiring an authentication token:
You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb)
You can generate your own latents to reproduce results, or tweak your prompt on a specific result you liked. [This notebook](https://github.com/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb) shows how to do it step by step. You can also run it in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb).
You can look out for [issues](https://github.com/huggingface/diffusers/issues) you'd like to tackle to contribute to the library.
You need to accept the model license before downloading or using the Stable Diffusion weights. Please, visit the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), read the license carefully and tick the checkbox if you agree. You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section](https://huggingface.co/docs/hub/security-tokens) of the documentation.
🤗 Diffusers provides pretrained diffusion models across multiple modalities, such as vision and audio, and serves
