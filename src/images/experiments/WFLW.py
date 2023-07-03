from src.images.configs import Config

pip_32_16_60_r101_l2_l1_10_1_nb10 = Config(**{
    "det_head": 'pip',
    "net_stride": 32,
    "batch_size": 16,
    "init_lr": 0.0001,
    "num_epochs": 60,
    "decay_steps": [30, 50],
    "input_size": 256,
    "backbone": 'resnet101',
    "pretrained": True,
    "criterion_cls": 'l2',
    "criterion_reg": 'l1',
    "cls_loss_weight": 10,
    "reg_loss_weight": 1,
    "num_lms": 98,
    "num_nb": 10,
    "use_gpu": True,
    "gpu_id": 3,
    "device": 'cuda',
    "data_name": 'WFLW',
    "experiment_name": 'pip_32_16_60_r101_l2_l1_10_1_nb10',
    "norm_points": [60, 72]
})
