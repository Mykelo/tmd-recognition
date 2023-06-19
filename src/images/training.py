from typing import Optional
import pytorch_lightning as pl
from PIPNet_lib.functions import *
from PIPNet_lib.networks import *
import torchvision.models as models
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch
from typing import Optional
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import src.data_modules as data_modules
from .types import Config
sys.path.insert(0, '..')

package_directory = os.path.dirname(os.path.abspath(__file__))


class PIPModule(pl.LightningModule):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=cfg.pretrained)
            net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms,
                               input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=cfg.pretrained)
            net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms,
                               input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'resnet101':
            resnet101 = models.resnet101(pretrained=cfg.pretrained)
            net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms,
                                input_size=cfg.input_size, net_stride=cfg.net_stride)
        else:
            raise Exception('No such backbone!')

        self.criterion_cls = None
        if cfg.criterion_cls == 'l2':
            self.criterion_cls = nn.MSELoss()
        elif cfg.criterion_cls == 'l1':
            self.criterion_cls = nn.L1Loss()
        else:
            raise Exception('No such cls criterion:', cfg.criterion_cls)

        self.criterion_reg = None
        if cfg.criterion_reg == 'l1':
            self.criterion_reg = nn.L1Loss()
        elif cfg.criterion_reg == 'l2':
            self.criterion_reg = nn.MSELoss()
        else:
            raise Exception('No such reg criterion:', cfg.criterion_reg)

        self.net = net

    def load_weights(self, path: str):
        if self.cfg.use_gpu:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        state_dict = torch.load(path, map_location=device)
        self.net.load_state_dict(state_dict)

    def update_config(self, new_cfg: Config):
        if new_cfg.backbone != self.cfg.backbone or new_cfg.det_head != self.cfg.det_head:
            raise Exception(
                'Given config files have incompatible network architectures')

        self.cfg = new_cfg
        self.net.cls_layer = nn.Conv2d(
            2048, new_cfg.num_lms, kernel_size=1, stride=1, padding=0)
        self.net.x_layer = nn.Conv2d(
            2048, new_cfg.num_lms, kernel_size=1, stride=1, padding=0)
        self.net.y_layer = nn.Conv2d(
            2048, new_cfg.num_lms, kernel_size=1, stride=1, padding=0)
        self.net.nb_x_layer = nn.Conv2d(
            2048, new_cfg.num_nb*new_cfg.num_lms, kernel_size=1, stride=1, padding=0)
        self.net.nb_y_layer = nn.Conv2d(
            2048, new_cfg.num_nb*new_cfg.num_lms, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        stats = self._common_step(batch, batch_idx, 'train')
        return stats['train_total_loss']

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        if self.cfg.pretrained:
            optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.init_lr)
        else:
            optimizer = optim.Adam(
                self.net.parameters(), lr=self.cfg.init_lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.decay_steps, gamma=0.1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def _common_step(self, batch, batch_idx, prefix):
        inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y = batch
        outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = self(
            inputs)
        loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = compute_loss_pip(
            outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, self.criterion_cls, self.criterion_reg, self.cfg.num_nb)

        loss_map_w = self.cfg.cls_loss_weight*loss_map
        loss_x_w = self.cfg.reg_loss_weight*loss_x
        loss_y_w = self.cfg.reg_loss_weight*loss_y
        loss_nb_x_w = self.cfg.reg_loss_weight*loss_nb_x
        loss_nb_y_w = self.cfg.reg_loss_weight*loss_nb_y
        loss = loss_map_w + loss_x_w + loss_y_w + loss_nb_x_w + loss_nb_y_w

        stats_dict = {
            f'{prefix}_total_loss': loss,
            f'{prefix}_map_loss': loss_map_w,
            f'{prefix}_x_loss': loss_x_w,
            f'{prefix}_y_loss': loss_y_w,
            f'{prefix}_nbx_loss': loss_nb_x_w,
            f'{prefix}_nby_loss': loss_nb_y_w
        }

        self.log_dict(stats_dict, prog_bar=True, on_step=False, on_epoch=True)
        return stats_dict


def pl_train(model: pl.LightningModule, cfg: Config, seed: Optional[int] = 42):
    dm = data_modules.LandmarksDataModule(cfg, seed)
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        dirpath=os.path.join('snapshots', cfg.data_name),
        filename=cfg.experiment_name,
        save_top_k=1,
        mode="min",
    )
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(
        'lightning_logs', cfg.data_name), name=cfg.experiment_name, log_graph=True)
    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        gpus=AVAIL_GPUS,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )
    trainer.fit(model, dm)
