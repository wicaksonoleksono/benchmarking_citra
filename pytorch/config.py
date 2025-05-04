from easydict import EasyDict
import torch

cfg = EasyDict({
    "device": "cuda",  # or "cpu"
    "num_epochs": 50,
    "data_params": {
        "batch_size": 32,
        "num_workers": 1,
        "data_path": "./data/aggregated/",
        "test_split": 0.2
    },
    "model": {
        "num_heads": 10,
        "freeze_backbone": True,
    },
    "optimizer_class": torch.optim.AdamW,
    "optimizer_name": "ADAM",
    "optimizer_params": {
        "lr": 5e-4,
        "weight_decay": 1e-6,
    },
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_params": {
        "T_max": 25,

    },
    "base_output_dir": "./runs",
    "do_test": True,
    "test_name": "final_eval",
    "model_names": [
        "mobilenetv3_small_100.lamb_in1k",
        "mobilenetv3_large_100.ra_in1k",
        "mobilenetv2_100.ra_in1k"
    ]
})
