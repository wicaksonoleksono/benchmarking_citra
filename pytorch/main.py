# run_multiple.py

import argparse
import importlib.util
import re
import logging
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
    device = cfg.device if torch.cuda.is_available() else "cpu"
    full_name = cfg.model_names[0] if isinstance(cfg.model_names, list) else model_name
    train_loader, val_loader, test_loader, _ = get_dataloaders(**cfg.data_params, model_name=full_name,
                                                               auto_transform=True)
    num_heads = cfg.model.num_heads
    freeze_backbone = cfg.model.freeze_backbone
    model = init_model(
        num_heads=num_heads,
        model_name=model_name,
        freeze_backbone=freeze_backbone
    )
    optimizer = cfg.optimizer_class(model.parameters(), **cfg.optimizer_params)
    lr_scheduler = cfg.scheduler_class(optimizer, **cfg.scheduler_params)
    ce_fn = nn.CrossEntropyLoss()
    metrics = Metrics()
    base_out = Path(cfg.base_output_dir)
    run_name = f"{model_name}.{cfg.optimizer_name}"
    out_dir = base_out / run_name
    tracker = Tracker(str(out_dir))
    start_epoch = 1
    latest_ckpt = tracker.get_latest_checkpoint()
    if latest_ckpt:
        match = re.search(r'epoch_?(\d+)', latest_ckpt)
        if match:
            checkpoint_epoch = int(match.group(1))
            start_epoch = checkpoint_epoch + 1
        else:
            checkpoint_epoch = 0
        model, optimizer, checkpoint_epoch, lr_scheduler = tracker.load_model(
            latest_ckpt, model, optimizer, lr_scheduler
        )
        logging.info(f"✅ Resuming from epoch {checkpoint_epoch} (training from {start_epoch})")
    else:
        logging.info("⭐ No checkpoints found - starting from scratch")

    # Check if training is already completed
    if start_epoch > cfg.num_epochs:
        logging.info(f"⚠️ Training already completed (epoch {start_epoch-1}/{cfg.num_epochs})")

        # Run testing if needed and we have a best model
        if cfg.get("do_test", False) and tracker.history["best"]["path"] and not tracker.history.get("tested", False):
            # Load the best model for testing
            best_path = tracker.history["best"]["path"]
            model, _, _, _ = tracker.load_model(best_path, model)

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

            # Mark as tested
            tracker.history["tested"] = True
            tracker.save_history()

            # Visualize test results
            vis_dir = out_dir / "visualizations"
            visualizer = TrainingVisualizer(tracker.history)
            visualizer.plot_confusion_matrix(test_metrics, str(vis_dir))

            logging.info("💾 Resumed testing completed")
        elif tracker.history.get("tested", False):
            logging.info("✅ Testing already completed")
        return

    # --- Run training loop ---
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        logging.info(f"\n🚀 Epoch {epoch}/{cfg.num_epochs}")
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
        logging.info(f"💾 Saved checkpoint for epoch {epoch}")
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualizer = TrainingVisualizer(tracker.history)
    visualizer.plot_metrics(str(vis_dir))
    if cfg.get("do_test", False) and not tracker.history.get("tested", False):
        if tracker.history["best"]["path"]:
            best_path = tracker.history["best"]["path"]
            model, _, _, _ = tracker.load_model(best_path, model)

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

        # Mark as tested and save visualization
        tracker.history["tested"] = True
        tracker.save_history()
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
    tasks = []
    for cfg_file in args.configs:
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
