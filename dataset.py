import os
import cv2
import copy
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DHF1KDataset(Dataset):
    def __init__(self, parameter, model, multi_frame=0, alternate=1):
        self.path_data = os.path.join(parameter.root, model)
        self.len_snippet = parameter.temporal
        self.model = model
        self.multi_frame = multi_frame
        self.alternate = alternate
        self.image_width, self.image_height = parameter.image_width, parameter.image_height

        self.image_transform = transforms.Compose([
            transforms.Resize((parameter.image_height, parameter.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        if self.model == "train":
            self.video_names = os.listdir(self.path_data)
            self.list_num_frame = [len(os.listdir(os.path.join(self.path_data, name, 'images'))) for name in self.video_names]
        elif self.model == "val":
            self.list_num_frame = []
            for v in os.listdir(self.path_data):
                for i in range(0, len(os.listdir(os.path.join(self.path_data, v, 'images'))) - self.alternate*self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        if self.model == "train":
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, self.list_num_frame[idx] - self.alternate * self.len_snippet + 1)
        elif self.model == "val":
            (file_name, start_idx) = self.list_num_frame[idx]

        path_image = os.path.join(self.path_data, file_name, 'images')
        path_map = os.path.join(self.path_data, file_name, 'maps')

        clip_image = []
        clip_gt = []

        for i in range(self.len_snippet):
            image = Image.open(os.path.join(path_image, '%04d.png' % (start_idx + self.alternate * i + 1))).convert('RGB')

            gt = np.array(Image.open(os.path.join(path_map, '%04d.png' % (start_idx + self.alternate * i + 1))).convert('L'))
            gt = gt.astype('float')

            if self.model == "train":
                gt = cv2.resize(gt, (self.image_width, self.image_height))

            if np.max(gt) > 1.0:
                gt = gt / 255.0
            clip_gt.append(torch.FloatTensor(gt))

            clip_image.append(self.image_transform(image))

        clip_image = torch.FloatTensor(torch.stack(clip_image, dim=0))
        clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

        if self.multi_frame == 0:
            return clip_image, clip_gt[31]
        else:
            return clip_image, clip_gt


class Hollywood_UCFDataset(Dataset):
    def __init__(self, parameter, model, frame_no="last", multi_frame=0):
        self.path_data = os.path.join(parameter.root, model)
        self.model = model
        self.len_snippet = parameter.temporal
        self.frame_no = frame_no
        self.multi_frame = multi_frame
        self.image_width, self.image_height = parameter.image_width, parameter.image_height

        self.image_transform = transforms.Compose([
            transforms.Resize((parameter.image_height, parameter.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        if self.model == "training":
            self.video_names = os.listdir(self.path_data)
            self.list_num_frame = [len(os.listdir(os.path.join(self.path_data, name, 'images'))) for name in self.video_names]

        elif self.model == "testing":
            self.list_num_frame = []
            for v in os.listdir(self.path_data):
                for i in range(0, len(os.listdir(os.path.join(self.path_data, v, 'images'))) - self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i))
                if len(os.listdir(os.path.join(self.path_data, v, 'images'))) <= self.len_snippet:
                    self.list_num_frame.append((v, 0))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        if self.model == "training":
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, max(1, self.list_num_frame[idx] - self.len_snippet + 1))
        elif self.model == "testing":
            (file_name, start_idx) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, 'images')
        path_annt = os.path.join(self.path_data, file_name, 'maps')

        clip_img = []
        clip_gt = []

        list_clips = os.listdir(path_clip)
        list_clips.sort()
        list_sal_clips = os.listdir(path_annt)
        list_sal_clips.sort()

        if len(list_sal_clips) < self.len_snippet:
            temp = [list_clips[0] for _ in range(self.len_snippet - len(list_clips))]
            temp.extend(list_clips)
            list_clips = copy.deepcopy(temp)

            temp = [list_sal_clips[0] for _ in range(self.len_snippet - len(list_sal_clips))]
            temp.extend(list_sal_clips)
            list_sal_clips = copy.deepcopy(temp)

            assert len(list_sal_clips) == self.len_snippet and len(list_clips) == self.len_snippet

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, list_clips[start_idx + i])).convert('RGB')
            clip_img.append(self.image_transform(img))

            gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[start_idx + i])).convert('L'))
            gt = gt.astype('float')

            if self.model == "training":
                gt = cv2.resize(gt, (self.image_width, self.image_height))

            if np.max(gt) > 1.0:
                gt = gt / 255.0
            clip_gt.append(torch.FloatTensor(gt))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        if self.multi_frame == 0:
            gt = clip_gt[-1]
        else:
            gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

        return clip_img, gt




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Hollywood2')
    parser.add_argument("--root", type=str, default='/home/jinyingjie/dataset/Hollywood2')    # Hollywood2 DHF1k UCF

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--temporal", type=int, default=32)
    parameter = parser.parse_args()

    if parameter.dataset == "DHF1K":
        train_data = DHF1KDataset(parameter=parameter, model='train')
    else:
        train_data = Hollywood_UCFDataset(parameter=parameter, model='training')
        val_data = Hollywood_UCFDataset(parameter=parameter, model='testing')

    train_loader = DataLoader(train_data, batch_size=parameter.batch_size, num_workers=parameter.workers)
    print(parameter.dataset)
    for batch in train_loader:
        image, label = batch
        print(image.shape)
        print(label.shape)