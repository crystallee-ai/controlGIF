import sys

from tqdm import tqdm
sys.path.append('./core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import argparse
import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader
import torch
import torchvision.transforms as transforms
from decord._ffi.base import DECORDError



def extract_optical_flow(csv_path, output_dir, video_folder, sample_stride, sample_n_frames, sample_size):

    parser = argparse.ArgumentParser()  
    args = parser.parse_args()
    model = torch.nn.DataParallel(RAFT(args))
    state_dict = torch.load("/root/lh/RAFT-master/models/raft-things.pth")
    model.load_state_dict(state_dict)

    model = model.module
    model.to("cuda")
    model.eval()

    with open(csv_path, 'r') as csvfile:
        dataset = list(csv.DictReader(csvfile))
        length = len(dataset)
    video_folder    =  video_folder
    sample_stride   = sample_stride
    sample_n_frames = sample_n_frames

    pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((sample_size, sample_size)),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    with tqdm(total=length) as pbar:
        pbar.set_description("Steps")
        for idx in range(length):
            video_dict = dataset[idx]
            videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
            output_path = output_dir+f"/{videoid}.npy"
            video_dir = os.path.join(video_folder, f"{videoid}.mp4")
            if os.path.exists(output_path):
                print(f"{output_path} already exists, continue")
                pbar.update(1)
                continue
            try:
                video_reader = VideoReader(video_dir)
            except Exception as e:
                print(f"Error reading video at {video_dir}, error: {e}")
                pbar.update(1)
                continue
            video_length = len(video_reader)
            
            # if not os.path.exists(output_path):
            #     os.mkdir(output_path)

            clip_length = min(video_length, (sample_n_frames - 1) * sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)

            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader

            

            pixel_values = pixel_transforms(pixel_values) # shape [16,3,256,256]
            #----------------------------------------------
            flow_ls = []
            with torch.no_grad():
                padder = InputPadder(pixel_values[0].shape)
                for j in range(pixel_values.shape[0]-1):

                    image1, image2 = padder.pad(pixel_values[j], pixel_values[j+1])
                    image1 = image1.unsqueeze(0).cuda()
                    image2 = image2.unsqueeze(0).cuda()
                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                    # extra_channel = torch.ones((1, 1, flow_up.shape[2],flow_up.shape[3]))
                    # flow = torch.concatenate([flow_up.cpu(),extra_channel], dim=1).squeeze(0)
                    flow = flow_up.cpu().squeeze(0) # shape [2, 256, 256]

                    flow_ls.append(flow)
            flow_ls = np.array(flow_ls) # shape [15, 2, 256, 256]
            np.save(output_path, flow_ls)
            pbar.update(1)


extract_optical_flow("/root/lh/AnimateDiff-main/results_2M_val.csv",\
                     "/root/lh/AnimateDiff-main/dataset_opticalflow" ,\
                     "/root/lh/AnimateDiff-main/datasets", 4, 16, 256)