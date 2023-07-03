from src.images.configs import Config
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn.parallel
import torch
import numpy as np
from PIL import Image
from typing import Optional
import cv2
import os
import sys
from src.common.PIPNet_lib.functions import forward_pip, get_meanface
from insightface.app import FaceAnalysis

import src.images.training as training
from src.images.utils import draw_points, HiddenPrints
sys.path.insert(0, '..')


package_directory = os.path.dirname(os.path.abspath(__file__))


def get_face_detector(det_size: tuple[int, int] = (256, 256)) -> FaceAnalysis:
    """
    Initializes the face detector from the InsightFace library. 
    https://github.com/deepinsight/insightface/tree/master/model_zoo
    """

    # FaceAnalysis module prints a lot of info during initialization which is not very 
    # usable, so temporarily disable printing
    with HiddenPrints():
        app = FaceAnalysis(providers=[
                            'CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
        app.prepare(ctx_id=0, det_size=det_size)

    return app


class PIPNet_PL:
    def __init__(self, cfg: Config, snapshots_path: Optional[str] = 'snapshots'):
        self.cfg = cfg
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([transforms.Resize(
            (cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
        self.snapshots_path = snapshots_path
        self.model = self.init_net()
        self.update_config(cfg)
        self.detector = get_face_detector()

    def init_net(self):
        cfg = self.cfg

        path = os.path.join(self.snapshots_path, cfg.data_name,
                            f'{cfg.experiment_name}.ckpt')
        return training.PIPModule.load_from_checkpoint(path, cfg=cfg)

    def update_config(self, new_cfg: Config):
        self.cfg = new_cfg
        self.meanface_indices, self.reverse_index1, self.reverse_index2, self.max_len = get_meanface(
            os.path.join(package_directory, '..', 'data', self.cfg.data_name, 'meanface.txt'), self.cfg.num_nb)
        if self.cfg.use_gpu:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def get_landmarks(self, image: np.ndarray):
        cfg = self.cfg
        reverse_index1, reverse_index2 = self.reverse_index1, self.reverse_index2
        max_len = self.max_len

        self.model.eval()

        image_resized = cv2.resize(image, (cfg.input_size, cfg.input_size))
        inputs = Image.fromarray(
            image_resized[:, :, ::-1].astype('uint8'), 'RGB')
        inputs = self.preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(self.device)

        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(
            self.model, inputs, self.preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)

        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1,
                                 reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1,
                                 reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(
            torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(
            torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()

        return lms_pred_merge

    def analyze(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        image_height, image_width, _ = image.shape
        faces = self.detector.get(image)
        # Take the biggest face
        x1, y1, x2, y2 = max(
            faces, key=lambda face: face['bbox'][3] - face['bbox'][1])['bbox']
        
        # Make it square
        width = x2 - x1
        height = y2 - y1
        dim_diff = height - width
        width_fill = max(dim_diff, 0)
        height_fill = -min(dim_diff, 0)

        # Make it a little bit bigger
        scale = 0.1
        x1 -= width_fill/2 + width * scale/2
        y1 -= height_fill/2 + height * scale/2
        x2 += width_fill/2 + width * scale/2
        y2 += height_fill/2 + height * scale/2

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image_width-1)
        y2 = min(y2, image_height-1)

        image_crop = image[int(y1):int(y2), int(x1):int(x2), :]
        image_crop = cv2.resize(image_crop, (cfg.input_size, cfg.input_size))
        landmarks = self.get_landmarks(image_crop)
        return np.array([x1, y1, x2, y2]), landmarks

    def draw(self, image: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray, save_path: Optional[str] = None, crop: bool = False, draw_indexes: bool = False) -> np.ndarray:
        landmarks = landmarks.copy()
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        landmarks[::2] *= bbox_width
        landmarks[1::2] *= bbox_height
        if crop:
            image = image[int(y1):int(y2), int(x1):int(x2), :]
        else:
            landmarks[::2] += x1
            landmarks[1::2] += y1

        draw_points(image, landmarks, save_path=save_path, draw_indexes=draw_indexes)


