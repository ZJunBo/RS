import pdb
from torch.utils.data import Dataset, DataLoader
import glob
import os
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler, RandomSampler
from ever.api.data import CrossValSamplerGenerator
import numpy as np
import logging
from utils.tools import seed_worker
import torch
import torch.nn.functional as F
import random
from PIL import Image
import ever as er
logger = logging.getLogger(__name__)



LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)

def soft_pseudo_label_randomcrop(img, soft_label):
    # img为已经读取到的RGB图像， soft_label为读取到的numpy数据

    lpsoft = torch.from_numpy(soft_label)
    lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=(1024,1024), mode='bilinear', align_corners=True)[0]  # augmentations = transforms.Compose([
    w, h = (1024,1024)
    tw, th = (512, 512)
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    img = img.crop((x1, y1, x1 + tw, y1 + th))
    img = np.array(img)
    transforms = Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ])
    blob = transforms(image=img)
    img = blob['image']

    lpsoft = lpsoft[:, y1:y1 + th, x1:x1 + tw]

    return img, lpsoft

class LoveDA(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms


    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png')) #[['./LoveDA/Train/Rural/images_png/1323.png', './LoveDA/Train/Rural/images_png/473.png', ...]

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list] # ['1323.png', '473.png'...]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        if len(self.cls_filepath_list) > 0:
            mask = imread(self.cls_filepath_list[idx]).astype(np.long) -1
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']

            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)



class LoveDA2(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png')) #[['./LoveDA/Train/Rural/images_png/1323.png', './LoveDA/Train/Rural/images_png/473.png', ...]

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list] # ['1323.png', '473.png'...]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                fname = fname.replace('.png', '.npy')
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = Image.open(self.rgb_filepath_list[idx])
        if len(self.cls_filepath_list) > 0:
            mask = np.load(self.cls_filepath_list[idx])
            image, mask = soft_pseudo_label_randomcrop(image, mask)

            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)

class LoveDALoader2(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = LoveDA2(self.config.image_dir, self.config.mask_dir, self.config.transforms)

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(LoveDALoader2, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True,
                                       drop_last=True
                                       )
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            scale_size=None,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))

class LoveDALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = LoveDA(self.config.image_dir, self.config.mask_dir, self.config.transforms)

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(LoveDALoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True,
                                       drop_last=True
                                       )
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            scale_size=None,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))


class TestDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.rgb_filepath_list = []
        if isinstance(image_dir, list):
            for img_dir_path in zip(image_dir):
                self.batch_generate(img_dir_path)

        else:
            self.batch_generate(image_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir):
        image_dir = image_dir[0]
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        self.rgb_filepath_list += rgb_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        mask = np.zeros([512, 512])
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob['image']

        return dict(rgb=image, fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)


class TestLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = TestDataset(self.config.image_dir, self.config.transforms)

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(TestLoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True,
                                       drop_last=True
                                       )

    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=1,
            num_workers=4,
            scale_size=None,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))