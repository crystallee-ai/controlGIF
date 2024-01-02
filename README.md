# controlGIF

[**Image2Video**] Animate a given image with [animatediff](https://github.com/guoyww/AnimateDiff) and controlnet

[![Animation](http://img.youtube.com/vi/lwXb_cJai8w/0.jpg)](https://www.youtube.com/watch?v=lwXb_cJai8w)

<video width="320" height="240" controls>
  <source src="show.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
***Above is a video demo link, please click to get a demo presentation.***

## Setup environment

```python
conda create --name controlgif python=3.10
conda activate controlgif
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# torch's version is 1.13.1 you can find other ways to install torch(cuda11.7) in https://pytorch.org/get-started/previous-versions/
pip install -r requirements.txt
```

## Download checkpoints

1. git clone the [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) from huggingface in ./checkpoints （format of diffusers, rather than a single .ckpt or .safetensors)
2. download some personalized models from [civitai](https://civitai.com/) (What I use most frequently is [dreamshaper](https://civitai.com/models/4384/dreamshaper)) in ./checkpoints/base_models
3. download motion model from [here](https://huggingface.co/crishhh/animatediff_controlnet) in ./checkpoints/unet_temporal
4. download controlnet model from [here](https://huggingface.co/crishhh/animatediff_controlnet) in ./checkpoints/controlnet

## File structure example

```markdown
- checkpoints
  - base_models
  	- dreamshaper_8.safetensors
  - controlnet
    - controlnet_checkpoint.ckpt
  - stable-diffusion-v1-5
    - feature_extractor
    - safety_checker
    - scheduler
    - text_encoder
    - tokenizer
    - unet
    - vae
  - unet_temporal
    - motion_checkpoint_less_motion.ckpt
    - motion_checkpoint_more_motion.ckpt
```

## Run the webui

```python
conda activate controlgif
python app.py
```

I can run it in my RTX3090. If you erase the clip_interrogator module, you can run it in low VRAM like 16G or 12G.

## Some notes

This method may lead to bad results when receive some portraits. (# in my TODO list)

And same method for SDXL version is under development , stay tuned! 

## Contact

If you have some questions, please open an issue or send an email to me at crystallee0418@gmail.com. 

## Acknowledgments

The code in this repository is derived from [Animatediff](https://github.com/guoyww/AnimateDiff) and Diffusers.