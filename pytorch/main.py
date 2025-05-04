# run_multiple.py

import argparse
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn

from get_dataloader import get_dataloaders
from train_eval import train, evaluate
from util import Tracker, Metrics, TrainingVisualizer, set_seed

from model import init_model


def run_experiment(config_file: str, model_name: str):
    set_seed(42)
    spec = importlib.util.spec_from_file_location("config", config_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    cfg = getattr(cfg_mod, "cfg", None)
    if cfg is None:
        raise ValueError(f"Config file {config_file} must define 'cfg' EasyDict")

    # --- prepare data ---
    device = cfg.device if torch.cuda.is_available() else "cpu"
    full_name = cfg.model_names[0]
    base_name = full_name.split('_', 1)[0]
    train_loader, val_loader, test_loader, _ = get_dataloaders(**cfg.data_params, model_name=base_name,
                                                               auto_transform=True)

    # --- build model for this task ---
    num_heads = cfg.model.num_heads
    freeze_backbone = cfg.model.freeze_backbone
    model = init_model(
        num_heads=num_heads,
        model_name=model_name,
        freeze_backbone=freeze_backbone
    )

    # --- optimizer, scheduler, loss, metrics ---
    optimizer = cfg.optimizer_class(model.parameters(), **cfg.optimizer_params)
    lr_scheduler = cfg.scheduler_class(optimizer, **cfg.scheduler_params)
    ce_fn = nn.CrossEntropyLoss()
    metrics = Metrics()
    # --- tracker + resume support ---
    base_out = Path(cfg.base_output_dir)
    run_name = f"{model_name}.{cfg.optimizer_name}"
    out_dir = base_out / run_name
    tracker = Tracker(str(out_dir))
    latest_ckpt = tracker.latest_checkpoint()
    start_epoch = 1
    if latest_ckpt:
        resumed = tracker.load_checkpoint(latest_ckpt, model, optimizer, lr_scheduler)
        start_epoch = resumed + 1

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train(
            device=device,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tracker=tracker,
            metrics=metrics,
            ce_fn=ce_fn,
        )
    vis_dir = out_dir / "visualizations"
    visualizer = TrainingVisualizer(tracker.history)
    visualizer.plot_metrics(str(vis_dir))

    if cfg.get("do_test", False):
        test_metrics = evaluate(
            device=device,
            data_iter=test_loader,
            model=model,
            ce_fn=ce_fn,
            tracker=tracker,
            epoch=None,
            is_testing=True,
            test_name=cfg.get("test_name", "test"),
            output_path=str(out_dir),
        )
        visualizer.plot_confusion_matrix(test_metrics, str(vis_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", "-c",
        nargs="+",
        required=True,
        help="List of config .py files to run"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel experiments (default: one per model)"
    )
    args = parser.parse_args()

    # build (config_file, model_name) task list
    tasks = []
    for cfg_file in args.configs:
        # load cfg to inspect its model_names
        spec = importlib.util.spec_from_file_location("config", cfg_file)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        cfg = getattr(cfg_mod, "cfg", None)
        if not cfg or "model_names" not in cfg:
            raise ValueError(f"{cfg_file} needs a `model_names` list in cfg")
        for m in cfg.model_names:
            tasks.append((cfg_file, m))
    max_workers = args.workers or len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = []
        for cfg_file, model_name in tasks:
            futures.append(exe.submit(run_experiment, cfg_file, model_name))
        for future in futures:
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                print(f"Error in experiment: {e}")


if __name__ == "__main__":
    main()
