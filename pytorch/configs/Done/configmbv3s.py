from easydict import EasyDict
import torch
num_epochs = 30
cfg = EasyDict({
    "device": "cuda",
    "num_epochs": num_epochs,
    "data_params": {
        "batch_size": 32,
        "num_workers": 4,
        "data_path": "./data/content/structured_dataset/",
        "test_split": 0.2
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
        "T_max": 100,
    },
    "base_output_dir": "./runs",
    "do_test": True,
    "test_name": "final_eval",
    "model_names": [
        "mobilenetv3_small_100.lamb_in1k",
    ]
})
