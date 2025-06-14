from model import FM
from benchmarker import find_best_model_path, benchmark_onnx
import torch.nn as nn
import numpy as np
from util import Metrics
import os
import json
from torch import nn
import torch
no_deprecation_warning = True


def evaluate(
    device,
    data_iter,
    model,
    ce_fn,
    tracker,
    epoch=None,
    is_testing=False,
    test_name=None,
    output_path=None,
    optimizer=None,
    train_metrics=None,
    train_loss=None,
):
    model.to(device).eval()
    metrics = Metrics()
    total_loss = 0.0
    is_arcface_model = isinstance(model, FM)
    name = getattr(model, "model_name", model.__class__.__name__)
    if is_arcface_model:
        print(f"Running evaluation in ArcFace mode for model: {name}")
    else:
        print(f"Running evaluation in Standard mode for model: {name}")
    if is_testing:
        os.makedirs(output_path, exist_ok=True)
        all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_iter):
            images = images.to(device)
            labels = labels.to(device)

            if is_arcface_model:
                logits = model.get_logits(images)
            else:
                logits = model(images)
            loss = ce_fn(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            metrics.update(preds.detach(), labels.detach())
            if is_testing:
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            if (batch_idx + 1) % 200 == 0:
                print(f"[Eval] Model ({name}) reached batch {batch_idx + 1} with loss: {loss.item():.4f}")
    avg_loss = total_loss / len(data_iter)
    final_metrics = metrics.compute()
    if is_testing:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics.true_labels = all_labels
        metrics.pred_labels = all_preds
        try:
            metrics.class_names = data_iter.dataset.classes
        except (AttributeError, KeyError):
            metrics.class_names = [f"Class {i}" for i in range(len(np.unique(all_labels)))]
        results = {**final_metrics, "loss": avg_loss}
        best_model_path = find_best_model_path(output_path)
        if best_model_path:
            print(f"Found best model: {best_model_path}")
            try:
                checkpoint = torch.load(best_model_path)
                if 'model' in checkpoint:
                    best_model = checkpoint['model']
                else:
                    best_model = model
                    best_model.load_state_dict(checkpoint['model_state'])
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using current model instance for benchmarking instead.")
                best_model = model

            best_model.to(device).eval()
            if is_arcface_model:
                best_model = best_model.backbone
            else:
                best_model = best_model
            try:
                onnx_results = benchmark_onnx(best_model, data_iter, device, output_path, test_name="LMP_2019")
                results.update(onnx_results)
            except Exception as e:
                print(f"Error during ONNX benchmarking: {e}")
        tracker.history["tested"] = True
        tracker.save_history()  # Penting: Simpan perubahan ke file history.json
        with open(os.path.join(output_path, f"{test_name}_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("\n📊 Final Test Results:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k.capitalize()}: {v:.4f}")
            else:
                print(f"{k.capitalize()}: {v}")
        return metrics

    tracker.update(
        epoch,
        loss=avg_loss,
        metrics={
            "accuracy":     final_metrics["accuracy"],
            "f1_macro":     final_metrics["f1_macro"],
            "f1_weighted":  final_metrics["f1_weighted"],
            "precision":    final_metrics["precision"],
            "recall":       final_metrics["recall"],

        },
        valid=True
    )
    tracker.save_best(model, optimizer, epoch, avg_loss, train_loss)
    tracker.save_history()
    return avg_loss, final_metrics


def train(
    device,
    epoch,
    train_loader,
    val_loader,
    model,
    optimizer,
    lr_scheduler,
    tracker,
    metrics,
    ce_fn: nn.Module = None,  # Make CrossEntropy loss optional
):

    model.to(device).train()
    metrics.reset()
    total_loss = 0.0
    name = getattr(model, "model_name", model.__class__.__name__)
    is_arcface_model = isinstance(model, FM)
    if is_arcface_model:
        print(f"Running in ArcFace mode for model: {name}")
    else:
        print(f"Running in Standard Classification mode for model: {name}")
        if ce_fn is None:
            raise ValueError("ce_fn (loss function) must be provided for standard models.")
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        if is_arcface_model:
            loss = model(images, labels)
            with torch.no_grad():
                logits = model.get_logits(images)
        else:
            logits = model(images)
            loss = ce_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        metrics.update(preds.detach(), labels.detach())

        if (batch_idx + 1) % 200 == 0:
            print(f"[Train] Model ({name}) reached batch {batch_idx + 1} with loss: {loss.item():.4f}")

        tracker.update(epoch, loss=loss.item(), valid=False)
    train_stats = metrics.compute()
    avg_train_loss = total_loss / len(train_loader)
    train_metrics = {
        "accuracy":     train_stats["accuracy"],
        "f1_macro":     train_stats["f1_macro"],
        "f1_weighted":  train_stats["f1_weighted"],
        "precision":    train_stats["precision"],
        "recall":       train_stats["recall"],
    }
    tracker.update(epoch,
                   loss=avg_train_loss,
                   metrics=train_metrics,
                   valid=False)
    avg_val_loss, val_stats = evaluate(
        device=device,
        epoch=epoch,
        data_iter=val_loader,
        model=model,
        ce_fn=ce_fn,
        tracker=tracker,
        is_testing=False,
        optimizer=optimizer,
        train_metrics=train_metrics,
        train_loss=avg_train_loss
    )
    print(
        f"Epoch {epoch} done. "
        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
        f"Train Acc: {train_stats['accuracy']:.2%} | Val Acc: {val_stats['accuracy']:.2%}\n"
        f"Train F1: {train_stats['f1_macro']:.2%} | Val F1: {val_stats['f1_macro']:.2%}\n"
        f"Train Recall: {train_stats['recall']:.2%} | Val Recall: {val_stats['recall']:.2%}"

    )
    tracker.save_checkpoint(model, optimizer, lr_scheduler, epoch)
    return val_stats["f1_macro"]
