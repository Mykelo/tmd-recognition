from typing import Optional
from pydantic import BaseModel


class Config(BaseModel):
    det_head: str
    net_stride: int
    batch_size: int
    init_lr: float
    num_epochs: int
    decay_steps: list[int]
    input_size: int
    backbone: str
    pretrained: bool
    criterion_cls: str
    criterion_reg: str
    cls_loss_weight: int
    reg_loss_weight: int
    num_lms: int
    num_nb: int
    use_gpu: bool
    gpu_id: int
    device: Optional[str]
    data_name: str
    experiment_name: str
    norm_points: Optional[list[int]]
