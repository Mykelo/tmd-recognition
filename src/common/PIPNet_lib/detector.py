import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from .networks import *
from .functions import *
from .experiments.WFLW.pip_32_16_60_r101_l2_l1_10_1_nb10 import Config


package_directory = os.path.dirname(os.path.abspath(__file__))


class PIPNet:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
        
        self.net = self.init_net()

    def init_net(self):
        cfg = self.cfg
        if cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=cfg.pretrained)
            net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=cfg.pretrained)
            net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'resnet101':
            resnet101 = models.resnet101(pretrained=cfg.pretrained)
            net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        else:
            print('No such backbone!')
            exit(0)

        if cfg.use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        net = net.to(device)

        weight_file = os.path.join(package_directory, 'snapshots', 'epoch%d.pth' % (cfg.num_epochs-1))
        state_dict = torch.load(weight_file, map_location=device)
        net.load_state_dict(state_dict)

        return net

    def get_landmarks(self, image):
        cfg = self.cfg
        meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join(package_directory, 'data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

        self.net.eval()

        image_resized = cv2.resize(image, (cfg.input_size, cfg.input_size))
        inputs = Image.fromarray(image_resized[:,:,::-1].astype('uint8'), 'RGB')
        inputs = self.preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(cfg.device)

        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(self.net, inputs, self.preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)

        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        
        return lms_pred_merge
    