from typing import Literal, Optional
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from collections.abc import Callable
import numpy as np
# import src.features as feat
import src.images.utils as utils
from sklearn.model_selection import train_test_split 
import pickle
import copy
import csv
import datetime
import pytz

def get_dynamic_features(path: str) -> dict[str, np.ndarray]:
    dynamic_features_df = pd.read_csv(path)
    columns_to_keep = dynamic_features_df.columns[:-3]
    dynamic_features_dict = {}
    for _, row in dynamic_features_df.iterrows():
        dynamic_features_dict[row['patient_id']] = row[columns_to_keep].to_numpy()
    return dynamic_features_dict

def get_denta_labels(
    path: str, 
    types: list[str] = ['photos.frontal.neutral', 'photos.frontal.open', 'photos.frontal.smile'], 
    split_type: Literal['random', 'date'] = 'random',
    random_state = 41,
    split_date: datetime.datetime = datetime.datetime(2022, 7, 14, tzinfo=pytz.UTC),
    filter_ids: bool = False
    ) -> pd.DataFrame:

    glasses, tilted = [], []
    if filter_ids:
        glasses = [
            '3d27f27c66338916830ff47a525a5ab670957ae187f9e179349706ab908ca028',
            '8e9bce05deefc84833608b33ea896e7c3c836c09956938d8fe3bb34a7bef15b3',
            '67cf7dedd626321582c620e5205086fc8bc2d5f9de31b575389f8ecf60e8ba50',
            '8611ad892db2db0401c899778132d56f1ffee75ece9576483e46f55227791311',
            '0558695ddf7ffae75270af0e1844815b498d91b934c54eda302dc08e405ae787',
            'c65f40509b007954ef6beb562db30b4d501ef2803dfd52c040b47f0c1b8c595d',
            'c26f41b477a9ce3fafeaddeb8397577af9dbddda0a82498768082b845bc07e2e',
        ]

        tilted = [
            '4ffd2cba7b1630a20c8aa376e20bd3be5db8eb8d40e84b5e57558e01ba8a839f', 
            '50b01c8cedc444aa826d3ba9dce882b1f73dddb3ab0c8231fe26623297dbf8d7',
            '6313e0750a053702762bbce2a967037bed367babb7ab8b22e1c395e55e2b946c',
            # '93627cc8cdaefc0fba04498921448d18f3117fca31768ae0a308fecb977367b0',
            'a2f5e59f6411b677ded259d6029cc69d2250efb9e38097e8f3323a85d341db20',
            'c13b8634f19cf55f7ab1545a56b7a6434d79852280e44535917ea216db26560f',
            # 'd143675194c12c7797ed7ca6d2c1f6c81a38b69ea975dc88c69f26b932c5dac9',
            'e7c4830c778321fd49eb3aa04b0d8fa5eb1d3161f796cd79491679e1c925a29d',
            '3df4df53c23d9247c8f56804bba55f4872fb6dda1e2b78c158ad13202987aa38',
        ]
    
    rows = []
    with open(os.path.join(path, 'data.csv'), newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for t in types:
                row_c = row.copy()
                row_c['Type'] = t
                # Make it compatible with datasets
                row_c['GroupID'] = row_c['Id']
                row_c['File'] = f'{row_c["Id"]}.{t}.jpg'
                row_c['Label'] = 0 if row_c['Diagnosis'] == 'healthy' else 1
                rows.append(row_c)
    df = pd.DataFrame(rows)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.loc[df['Id'].isin(glasses), 'Diagnosis'] = 'glasses'
    df.loc[df['Id'].isin(tilted), 'Diagnosis'] = 'tilted'
    df = df[(df['Diagnosis'] == 'healthy') | (df['Diagnosis'] == 'sick')]

    if split_type == 'random':
        df_grouped = df.groupby(by='GroupID').first().reset_index(drop=True)
        train_ids, test_ids = train_test_split(df_grouped['Id'], test_size=0.2, random_state=random_state, stratify=df_grouped['Label'])

        df.loc[df['Id'].isin(train_ids), 'Split'] = 'training'
        df.loc[df['Id'].isin(test_ids), 'Split'] = 'test'
    elif split_type == 'date':
        df.loc[df['Timestamp'] < split_date, 'Split'] = 'training'
        df.loc[df['Timestamp'] >= split_date, 'Split'] = 'test'
    return df



def flip_image(image: np.ndarray, target: np.ndarray, points_flip: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.flip(image, 1)
    target = np.array(target).reshape(-1, 2)
    if points_flip is not None:
        target = target[points_flip, :]
    target[:, 0] = 1-target[:, 0]
    target = target.flatten()
    return image, target


class ImageDataset(Dataset):
    def __init__(self, root: str, labels: pd.DataFrame, transform=None, target_transform=None):
        self.root = root
        self.labels = labels
        self.unique_groups = labels['GroupID'].unique()
        self.unique_groups.sort()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file = self.labels.iloc[idx]['File']
        img_path = os.path.join(
            self.root, 'processed_images', file)
        image = cv2.imread(img_path)
        label = self.labels.iloc[idx]['Label']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return file, image, label


class LandmarkFeaturesWithDetectionDataset(Dataset):
    def __init__(self, imageDataset: ImageDataset,
                 landmarks_extractor: Callable[[np.ndarray], np.ndarray],
                 features_extractor,
                 prepare: bool = True
                 ):

        self.imageDataset = imageDataset
        self.landmarks_extractor = landmarks_extractor
        self.features_extractor = features_extractor
        self.landmarks = None
        self.angles = None
        self.features = None
        self.targets = None

        if prepare:
            landmarks, angles, features, targets = [], [], [], []
            for i in tqdm(range(len(imageDataset))):
                l, a, f, t = self._calc_features(i)
                landmarks.append(l)
                angles.append(a)
                features.append(f)
                targets.append(t)
            self.landmarks = np.array(landmarks)
            self.angles = np.array(angles)
            self.features = np.array(features)
            self.targets = np.array(targets)

    def _calc_features(self, idx):
        _, image, label = self.imageDataset[idx]
        points = self.landmarks_extractor(image)
        points, an, features = self.features_extractor.calculate(image, points)

        return points, an, features, label

    def __len__(self):
        return len(self.imageDataset)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def get_landmarks(self, idx: int) -> np.ndarray:
        return self.landmarks[idx], self.angles[idx]

    def __copy__(self) -> Dataset:
        dataset_copy = LandmarkFeaturesWithDetectionDataset(
            self.imageDataset, self.landmarks_extractor, self.features_extractor, False)
        dataset_copy.landmarks = self.landmarks.copy()
        dataset_copy.angles = self.angles.copy()
        dataset_copy.features = self.features.copy()
        dataset_copy.targets = self.targets.copy()
        return dataset_copy


class LandmarkFeaturesDataset(Dataset):
    def __init__(self, imageDataset: ImageDataset,
                 #  landmarks_extractor: Callable[[np.ndarray], np.ndarray],
                 features_extractor,
                 prepare: bool = True
                 ):

        self.imageDataset = imageDataset
        # self.landmarks_extractor = landmarks_extractor
        self.features_extractor = features_extractor
        self.landmarks = None
        self.angles = None
        self.features = None
        self.features_grouped = {}
        self.targets_grouped = {}
        self.files_grouped = {}
        self.targets = None
        # Load original landmarks, they should be in the same directory
        # as images
        with open(os.path.join(
                imageDataset.root, 'processed_images', 'landmarks.pkl'), 'rb') as handle:
            self.landmarks = pickle.load(handle)
        # self.landmarks = np.load(os.path.join(
        #     imageDataset.root, 'processed_images', 'landmarks.npy'))

        if prepare:
            landmarks, angles, features, targets = {}, {}, {}, {}
            for i in tqdm(range(len(imageDataset))):
                file, l, a, f, t = self._calc_features(i)
                landmarks[file] = l
                angles[file] = a
                features[file] = f
                targets[file] = t
            # Overwrite original landmarks with normalized ones (head tilt is corrected)
            self.landmarks = landmarks
            self.angles = angles
            self.features = features
            self.targets = targets

            # Group features by group id
            for i in range(len(imageDataset.unique_groups)):
                # We want to get features of all images from the same group
                group = self.imageDataset.unique_groups[i]
                df = self.imageDataset.labels
                files = df[df['GroupID'] == group]['File'].to_list()
                files = list(sorted(files))

                features = [self.features[f] for f in files]
                # Target should be the same for all images in a group
                target = self.targets[files[0]]
                self.features_grouped[group] = features
                self.targets_grouped[group] = target
                self.files_grouped[group] = files

    def _calc_features(self, idx):
        file, image, label = self.imageDataset[idx]
        points = self.landmarks[file]
        _, points, an, features = self.features_extractor.calculate(
            image, points)

        return file, points, an, features, label

    def __len__(self):
        return len(self.imageDataset.unique_groups)

    def __getitem__(self, idx):
        group = self.imageDataset.unique_groups[idx]
        features = self.features_grouped[group]
        target = self.targets_grouped[group]
        return features, target

    def get_landmarks(self, idx: int) -> np.ndarray:
        file, _, _ = self.imageDataset[idx]
        return self.landmarks[file], self.angles[file]

    def __copy__(self) -> Dataset:
        dataset_copy = LandmarkFeaturesDataset(
            self.imageDataset, self.features_extractor, False)
        dataset_copy.landmarks = copy.deepcopy(self.landmarks)
        dataset_copy.angles = copy.deepcopy(self.angles)
        dataset_copy.features = copy.deepcopy(self.features)
        dataset_copy.targets = copy.deepcopy(self.targets)
        return dataset_copy


class FacialLandmarksDataset(Dataset):
    image_names: list[str]
    data_name: str

    def __init__(self, root_dir: str, target_size: int = 256):
        self.root_dir = root_dir
        self.target_size = target_size

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index) -> list[np.ndarray, np.ndarray]:
        return super().__getitem__(index)

    def draw(self, i: int, draw_indexes: bool = False):
        image, anno = self[i]
        utils.draw_points(image, np.array(anno), save_path=os.path.join(
            f'../data/samples/{self.data_name}', f'{i}.jpg'), draw_indexes=draw_indexes)
