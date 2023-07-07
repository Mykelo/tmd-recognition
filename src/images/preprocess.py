from tqdm import tqdm
import pandas as pd
import cv2
import os
import numpy as np
from src.images.datasets import (
    FacialLandmarksDataset,
)
import pickle

from src.images.configs import Config
from src.images.experiments import WFLW
from src.images.detector import PIPNet_PL

package_directory = os.path.dirname(os.path.abspath(__file__))


def crop_image(
    image: np.ndarray, bbox: list[float], scale: float = 0.2
) -> tuple[np.ndarray, list[float]]:
    image_height, image_width, _ = image.shape

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    dim_diff = height - width
    width_fill = max(dim_diff, 0)
    height_fill = -min(dim_diff, 0)

    x1 -= width_fill / 2 + width * scale / 2
    y1 -= height_fill / 2 + height * scale / 2
    x2 += width_fill / 2 + width * scale / 2
    y2 += height_fill / 2 + height * scale / 2

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image_width - 1)
    y2 = min(y2, image_height - 1)

    image_crop = image[int(y1) : int(y2), int(x1) : int(x2), :]
    return image_crop, [x1, y1, x2, y2]


def extract_faces_denta(
    path: str, labels: pd.DataFrame, detector=None, target_size: int = 256
):
    """
    Extracts faces from the Denta dataset and saves them in processed_images/ folder.
    """
    images_target_path = os.path.join(path, "processed_images")
    if not os.path.exists(images_target_path):
        os.mkdir(images_target_path)

    for _, row in tqdm(labels.iterrows(), total=labels.shape[0]):
        image_path = os.path.join(path, row["Id"], row["Type"] + ".jpg")
        image = cv2.imread(image_path)
        faces = detector(image)
        bbox = max(faces, key=lambda face: face["bbox"][3] - face["bbox"][1])["bbox"]

        image_crop, _ = crop_image(image, bbox, scale=0.1)
        image_crop = cv2.resize(image_crop, (target_size, target_size))

        cv2.imwrite(os.path.join(images_target_path, row["File"]), image_crop)


def extract_landmarks(
    path: str,
    labels: pd.DataFrame,
    cfg: Config = WFLW.pip_32_16_60_r101_l2_l1_10_1_nb10,
    snapshots_path: str = os.path.join(package_directory, "..", "..", "data", "snapshots")
):
    images_target_path = os.path.join(path, "processed_images")
    if not os.path.exists(images_target_path):
        raise Exception(
            "processed_images/ directory not found. You should call the extract_faces_tnf() function first."
        )

    pipnet = PIPNet_PL(
        cfg,
        snapshots_path=snapshots_path,
    )

    all_landmarks = {}
    for _, row in tqdm(labels.iterrows(), total=labels.shape[0]):
        img_path = os.path.join(path, "processed_images", row["File"])

        image = cv2.imread(img_path)
        landmarks = pipnet.get_landmarks(image)
        all_landmarks[row["File"]] = landmarks

    with open(os.path.join(images_target_path, "landmarks.pkl"), "wb") as handle:
        pickle.dump(all_landmarks, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_data(
    dataset: FacialLandmarksDataset,
    indices: list[int],
    root_path: str,
    anno_file: str,
    img_dir: str,
):
    image_names = dataset.image_names
    with open(os.path.join(root_path, anno_file), "w") as f:
        for i in tqdm(indices):
            img, anno = dataset[i]
            img_name = image_names[i]
            cv2.imwrite(os.path.join(root_path, img_dir, img_name), img)
            f.write(img_name + " ")
            for x, y in anno.reshape((-1, 2)):
                f.write(str(x) + " " + str(y) + " ")
            f.write("\n")
