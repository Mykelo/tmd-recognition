from typing import Optional
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import pandas as pd
import copy
import torchvision.transforms as transforms
from PIPNet_lib.data_utils import ImageFolder_pip
from PIPNet_lib.functions import get_label, get_meanface
import src.datasets as datasets
import src.features as feat
from .experiments import TNF, WFLW
import src.detector as detector
from .types import Config
import torch
import os
import numpy as np


package_directory = os.path.dirname(os.path.abspath(__file__))


class LandmarksDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config, seed: int or None):
        super().__init__()
        self.cfg: Config = cfg
        self.img_train: ImageFolder_pip or None = None
        self.img_val: ImageFolder_pip or None = None
        self.img_test: ImageFolder_pip or None = None
        self.seed = seed

    def prepare_data(self):
        cfg = self.cfg
        self.meanface_indices, _, _, _ = get_meanface(os.path.join(
            package_directory, '..', 'data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

        points_flip = None
        if cfg.data_name == 'data_300W':
            points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35,
                           34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
            points_flip = (np.array(points_flip)-1).tolist()
            assert len(points_flip) == 68
        elif cfg.data_name == 'WFLW':
            points_flip = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40,
                           39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
            assert len(points_flip) == 98
        elif cfg.data_name == 'COFW':
            points_flip = [2, 1, 4, 3, 7, 8, 5, 6, 10, 9, 12, 11, 15, 16,
                           13, 14, 18, 17, 20, 19, 21, 22, 24, 23, 25, 26, 27, 28, 29]
            points_flip = (np.array(points_flip)-1).tolist()
            assert len(points_flip) == 29
        elif cfg.data_name == 'AFLW':
            points_flip = [6, 5, 4, 3, 2, 1, 12, 11,
                           10, 9, 8, 7, 15, 14, 13, 18, 17, 16, 19]
            points_flip = (np.array(points_flip)-1).tolist()
            assert len(points_flip) == 19
        elif cfg.data_name == 'TNF':
            points_flip = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33,
                           32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]
            points_flip = (np.array(points_flip)-1).tolist()
            assert len(points_flip) == 68
        elif cfg.data_name == 'MENPO_profile_both':
            points_flip = list(range(39))
            points_flip = (np.array(points_flip)-1).tolist()
            assert len(points_flip) == 39
        elif cfg.data_name == 'MENPO_profile_left' or cfg.data_name == 'MENPO_profile_right':
            # We don't want to flip images if they are already flipped
            points_flip = None
        elif cfg.data_name == 'FITYMI':
            points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35,
                           34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66, 69, 68]
            points_flip = (np.array(points_flip)-1).tolist()
            assert len(points_flip) == 70
        else:
            raise Exception('No such data!')

        self.points_flip = points_flip

    def setup(self, stage: Optional[str] = None):
        cfg = self.cfg
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        labels_train = get_label(cfg.data_name, 'train.txt')
        self.img_train = ImageFolder_pip(os.path.join(package_directory, '..', 'data', cfg.data_name, 'images_train'),
                                         labels_train, cfg.input_size, cfg.num_lms,
                                         cfg.net_stride, self.points_flip, self.meanface_indices,
                                         transforms.Compose([
                                             transforms.RandomGrayscale(0.2),
                                             transforms.ToTensor(),
                                             normalize]))

        labels_test = get_label(cfg.data_name, 'test.txt')
        self.img_test = ImageFolder_pip(os.path.join(package_directory, '..', 'data', cfg.data_name, 'images_test'),
                                        labels_test, cfg.input_size, cfg.num_lms,
                                        cfg.net_stride, self.points_flip, self.meanface_indices,
                                        transforms.Compose([
                                            transforms.RandomGrayscale(0.2),
                                            transforms.ToTensor(),
                                            normalize]))
        self.img_val = ImageFolder_pip(os.path.join(package_directory, '..', 'data', cfg.data_name, 'images_test'),
                                       labels_test, cfg.input_size, cfg.num_lms,
                                       cfg.net_stride, self.points_flip, self.meanface_indices,
                                       transforms.Compose([
                                           transforms.RandomGrayscale(0.2),
                                           transforms.ToTensor(),
                                           normalize]))

    def train_dataloader(self):
        return DataLoader(self.img_train, batch_size=self.cfg.batch_size, drop_last=True, shuffle=True, num_workers=8, pin_memory=self.cfg.device == 'cuda')

    def val_dataloader(self):
        return DataLoader(self.img_val, batch_size=self.cfg.batch_size, drop_last=True, num_workers=8, pin_memory=self.cfg.device == 'cuda')

    def test_dataloader(self):
        return DataLoader(self.img_test, batch_size=self.cfg.batch_size, drop_last=True, num_workers=8, pin_memory=self.cfg.device == 'cuda')


class DentaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        seed: Optional[int] = None,
        cfg: Config = WFLW.pip_32_16_60_r101_l2_l1_10_1_nb10,
        features_extractor: feat.FacialFeaturesExtractor = feat.DominguezFeaturesExtractor(),
        df: Optional[pd.DataFrame] = None
    ):
        super().__init__()
        self.data_dir: str = data_dir
        self.metadata: pd.DataFrame or None = None
        self.data: datasets.LandmarkFeaturesDataset or None = None
        self.tnf_train: datasets.LandmarkFeaturesDataset or None = None
        self.tnf_val: datasets.LandmarkFeaturesDataset or None = None
        self.tnf_test: datasets.LandmarkFeaturesDataset or None = None
        self.seed = seed
        self.generator = None if seed is None else torch.Generator().manual_seed(self.seed)
        self.cfg = cfg
        self.features_extractor = features_extractor
        self.df = df

    def aggregate(self, avg_to: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        avg_to = self.features_extractor.geometric_features_num if avg_to is None else avg_to
        final_features = []
        for features in self.data.features_grouped.values():
            avgs = []
            concats = []
            for feat in features:
                avgs.append(feat[:avg_to])
                concats.append(feat[avg_to:])
            # concats = np.concatenate(concats)
            # We want to average neutral and smile photos
            # Photos with an open mouth should be used to calculate a difference
            # with the neutral one
            open_mouth = avgs[0] - avgs[1]
            avgs = np.stack([avgs[0], avgs[2]]).mean(axis=0)
            # final_features.append(np.concatenate([avgs, open_mouth, concats]))
            final_features.append(np.concatenate([avgs, concats[0], concats[2]]))
        return np.array(final_features), np.array(list(self.data.targets_grouped.values()))

    def filter_photos(self, photo_idx: int) -> tuple[np.ndarray, np.ndarray]:
        final_features = []
        for features in self.data.features_grouped.values():
            final_features.append(features[photo_idx])
        return np.array(final_features), np.array(list(self.data.targets_grouped.values()))

    def prepare_data(self):
        if self.df is None:
            self.df = datasets.get_denta_labels(self.data_dir)

        image_dataset = datasets.ImageDataset(self.data_dir, self.df)
        self.data = datasets.LandmarkFeaturesDataset(
            image_dataset, self.features_extractor)

    def setup(self, stage: Optional[str] = None):
        size = len(self.data)
        train_size = int(size * 0.9)
        self.tnf_train, self.tnf_val = random_split(
            self.data, [train_size, size - train_size], self.generator)

        self.tnf_test = copy.copy(self.tnf_val)

    def train_dataloader(self):
        return DataLoader(self.tnf_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.tnf_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.tnf_test, batch_size=32)