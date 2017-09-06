import os
import os.path
import glob
import numpy as np
import torch.utils.data as data
import torch as th
import time

from PIL import Image
from data_parser import JpegDataset
from transforms import ToTensor
from transforms import *

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class VideoFolder(data.Dataset):

    def __init__(self, root, json_file, clip_size, nclips, step_size, is_val,
                 transform=None, target_transform=None, loader=default_loader):
        json_data = KineticsDataset(json_file, root)
        classes, class_to_idx = json_data.classes, json_data.label2id
        videos = json_data.data2list()

        if len(videos) == 0:
            raise(RuntimeError("Found 0 video in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.videos = videos
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

        self.toTensor = ToTensor()

    def __getitem__(self, index):
        item = self.videos[index]
        path, target, target_idx = item.folder, item.label, item.label_idx

        img_paths = self.get_frame_names(path)
        imgs = []
        for img_path in img_paths:
            img = self.loader(img_path)
            # img = self.toTensor(img)

            if self.transform is not None:
                img = self.transform(img)
            imgs.append(th.unsqueeze(img, 0))

        if self.target_transform is not None:
            target_idx = self.target_transform(target_idx)

        # format data to torch
        data = th.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        return (data, target_idx)

    def __len__(self):
        return len(self.videos)

    def get_frame_names(self, path):
        # find video frames
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        # set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames:
            # Pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * \
                (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        frame_names = frame_names[offset:num_frames_necessary +
                                  offset:self.step_size]
        return frame_names


if __name__ == '__main__':
    transform = Compose([
                        CenterCrop(84),
                        ToTensor(),
                        # Normalize(
                        #     mean=[0.485, 0.456, 0.406],
                        #     std=[0.229, 0.224, 0.225])
                        ])
    loader = VideoFolder(root="/hdd/20bn-datasets/20bn-jester-v1/",
                         json_file="csv_files/jester-v1-validation.csv",
                         clip_size=18,
                         nclips=1,
                         step_size=2,
                         is_val=True,
                         transform=transform,
                         target_transform=None,
                         loader=default_loader)

    train_loader = th.utils.data.DataLoader(
        loader,
        batch_size=16, shuffle=True,
        num_workers=18, pin_memory=True)

    count = 0
    start = time.time()
    for i, a in enumerate(train_loader):
        if count == 100:
            break
        print(str(i), " -- ", str(a[0].size()))
        count += 1

    print(time.time() - start)
