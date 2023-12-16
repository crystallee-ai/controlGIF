# controlGIF
[**Image2Video**] Animate a given image with [animatediff](https://github.com/guoyww/AnimateDiff) and controlnet

[![Animation](http://img.youtube.com/vi/lwXb_cJai8w/0.jpg)](https://www.youtube.com/watch?v=lwXb_cJai8w)

<video width="320" height="240" controls>
  <source src="show.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Setup environment
```python
conda create --name controlgif python=3.10
conda activate controlgif
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# torch's version is 1.13.1 you can find other ways to install torch(cuda11.7) in https://pytorch.org/get-started/previous-versions/
pip install -r requirements.txt
```

## Download checkpoints
1. git clone the [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) from huggingface in ./checkpoints
2. download some personalized models from [civitai](https://civitai.com/) (What I use most frequently is [dreamshaper](https://civitai.com/models/4384/dreamshaper)) in ./checkpoints/base_models
3. download motion model from [here](https://huggingface.co/crishhh/animatediff_controlnet) in ./checkpoints/unet_temporal
4. download controlnet model from [here](https://huggingface.co/crishhh/animatediff_controlnet) in ./checkpoints/controlnet

## Run the webui
```python
conda activate controlgif
python app.py
```

I can run it in my RTX3090.

## Acknowledgments

The code in this repository is derived from [Animatediff](https://github.com/guoyww/AnimateDiff) and Diffusers.
