from easydict import EasyDict
import torch

cfg = EasyDict({
    # Device and training
    "device": "cuda",  # or "cpu"
    "num_epochs": 20,
    "data_params": {
        "batch_size": 32,
        "num_workers": 10,
    },
    "model": {
        "num_heads": 7,
    },
    "optimizer_class": torch.optim.AdamW,
    "optimizer_name": "ADAM",
    "optimizer_params": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "scheduler_class": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {
        "step_size": 5,
        "gamma": 0.1,
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
