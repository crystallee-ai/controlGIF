# controlGIF
Animate a given image with animatediff and controlnet

<video width="320" height="240" controls>
  <source src="show.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Setup environment
‘’‘Python
conda create --name controlgif python=3.10
pip install -r requirements.txt
'''
## Download checkpoints
1. git clone the stable-diffusion-v1-5 from huggingface in ./checkpoints
2. download some personalized models from civitai(What I use most frequently is dreamshaper) in ./checkpoints/base_models
3. download motion model from here in ./checkpoints/unet_temporal
4. download controlnet model from here in ./checkpoints/controlnet

## Run the webui
'''Python
python app.py
'''