from easydict import EasyDict
import torch
n = 50
cfg = EasyDict({
    "device": "cuda",
    "num_epochs": n,
    "data_params": {
        "batch_size": 32,
        "num_workers": 1,
        "data_path": "./nodup_data/",
        "test_split": 0.1,
        "val_split": 0.1,
    },
    "model": {
        "num_heads": 10,
        "freeze_backbone": False,
    },
    "optimizer_class": torch.optim.AdamW,
    "optimizer_name": "ADAM",
    "optimizer_params": {
        "lr": 1e-4,
        "weight_decay": 1e-7,
    },
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_params": {
        "T_max": n,
        "eta_min": 1e-6,
    },
    "base_output_dir": "./runs",
    "do_test": True,
    "test_name": "final_eval",
    "model_names": [
        "mnasnet_small.lamb_in1k",
    ]
})
