import argparse
import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader


import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print




class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder, opticalflow_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.opticalflow_folder = opticalflow_folder
        self.sample_stride   = sample_stride
        # self.sample_n_frames = sample_n_frames
        self.sample_n_frames = 16
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])


    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values,pixel_values[0], name
    
    def get_opticalflow(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        del video_reader

        opticalflow_dir = os.path.join(self.opticalflow_folder, f"{videoid}.npy")
        flow = np.load(opticalflow_dir)
        extra_channel = np.ones((flow.shape[0], 1, flow.shape[2],flow.shape[3]))
        flow = np.concatenate([flow,extra_channel], axis=1)
        flow = torch.from_numpy(flow).to(dtype=torch.float)
        # flow = flow / 255.
        return pixel_values, flow, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, image, name = self.get_batch(idx)
                # pixel_values, flow, name = self.get_opticalflow(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values) # shape [16,3,256,256]
        # init_image = pixel_values[0, :, :, :]
        sample = dict(pixel_values = pixel_values, image = pixel_values[0, :, :, :], text = name)
        # sample = dict(image = init_image, video = pixel_values, pixel_values = flow, text = name)
        return sample



if __name__ == "__main__":
    from animatediff.utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/results_2M_train.csv",
        video_folder="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/2M_val",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
