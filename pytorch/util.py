from pathlib import Path
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import pandas as pd
import os
import json
from torch import nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
import torch
logging.basicConfig(level=logging.INFO)

# written by @wicaksonolxn 02.11.24


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Metrics:
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []

    def reset(self):
        self.true_labels.clear()
        self.pred_labels.clear()

    def update(self, batch_preds, batch_labels):
        self.pred_labels.extend(batch_preds.cpu().numpy())
        self.true_labels.extend(batch_labels.cpu().numpy())

    def compute(self):
        return {
            "accuracy":  accuracy_score(self.true_labels, self.pred_labels),
            "precision": precision_score(self.true_labels, self.pred_labels, average='macro', zero_division=0),
            "recall":    recall_score(self.true_labels, self.pred_labels, average='macro', zero_division=0),
            "f1_macro":  f1_score(self.true_labels, self.pred_labels, average='macro', zero_division=0),
            "f1_weighted": f1_score(self.true_labels, self.pred_labels, average='weighted', zero_division=0)
        }


class Tracker:
    def __init__(self, output_path):
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "training_history.json"
        self.history = self._load_history()
        self.ckpt_pattern = "epoch_*.pth"

    def _load_history(self):
        if self.history_file.exists():
            return json.loads(self.history_file.read_text())
        # Initialize with recall as the metric to track for best model
        return {"train": {}, "valid": {}, "best": {"recall": -float('inf'), "epoch": None, "path": None}}

    def save_history(self):
        self.history_file.write_text(json.dumps(self.history, indent=2))

    def _cleanup(self, keep_names):
        for ckpt in self.output_dir.glob(self.ckpt_pattern):
            if ckpt.name not in keep_names:
                try:
                    ckpt.unlink()
                    logging.info(f"Removed checkpoint {ckpt.name}")
                except Exception as e:
                    logging.warning(f"Could not remove {ckpt.name}: {e}")

    def save_checkpoint(self, model, optimizer, lr_scheduler, epoch):
        fname = f"epoch_{epoch}.pth"
        keep = {fname, Path(self.history['best']['path']).name if self.history['best']['path'] else ''}
        self._cleanup(keep)
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'lr_scheduler_state': lr_scheduler.state_dict()
        }
        path = self.output_dir / fname
        torch.save(checkpoint, path)
        self.save_history()
        return path

    def load_checkpoint(self, path, model, optimizer=None, lr_scheduler=None, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logging.info(f"Loaded model from {path}")
        if optimizer and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            logging.info("Loaded optimizer state")
        if lr_scheduler and 'lr_scheduler_state' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state'])
            logging.info("Loaded LR scheduler state")
        self.history = self._load_history()
        return ckpt.get('epoch', 0)

    def update(self, epoch, loss=None, metrics=None, valid=False):
        section = 'valid' if valid else 'train'
        key = f"epoch_{epoch}"
        entry = self.history.setdefault(section, {}).setdefault(key, {})
        if loss is not None:
            entry['loss'] = loss
        if metrics:
            entry.update(metrics)
        self.save_history()

    def save_best(self, model, optimizer, epoch, recall):
        best = self.history['best']
        if recall <= best.get('recall', -float('inf')):
            return False
        if best['path']:
            try:
                Path(best['path']).unlink()
            except Exception:
                pass
        fname = f"best_epoch_{epoch}.pth"
        path = self.output_dir / fname
        torch.save({
            'epoch': epoch,
            'model': model,  # Save the entire model
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None
        }, path)
        self.history['best'] = {'recall': recall, 'epoch': epoch, 'path': str(path)}
        self.save_history()
        logging.info(f"New best model saved: {fname}")
        return True

    def latest_checkpoint(self):
        epochs = []
        for ckpt in self.output_dir.glob(self.ckpt_pattern):
            try:
                num = int(ckpt.stem.split('_')[1])
                epochs.append((num, ckpt))
            except Exception:
                continue
        if not epochs:
            return None
        _, latest = max(epochs, key=lambda x: x[0])
        return str(latest)


class TrainingVisualizer:
    def __init__(self, history):
        self.history = history
        self.metrics = ['loss', 'acc', 'precision', 'recall', 'f1_macro']
        self.labels = {
            'loss': 'Cross-Entropy Loss',
            'acc': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_macro': 'F1 Score (Macro)'
        }

    def plot_metrics(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        train_df, valid_df = self._prepare_epoch_data()
        for metric in self.metrics:
            plt.figure(figsize=(18, 12))

            if metric == 'loss':
                self._plot_loss(plt, train_df, valid_df)
            else:
                self._plot_standard_metric(plt, metric, train_df, valid_df)

            plot_path = os.path.join(output_path, f'{metric}_plot.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"📊 Saved {metric} plot to {plot_path}")

    def plot_confusion_matrix(self, metrics, output_path):
        """Plot using data from Metrics instance"""
        os.makedirs(output_path, exist_ok=True)
        cm = confusion_matrix(metrics.true_labels, metrics.pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=metrics.class_names if hasattr(metrics, 'class_names') else ['Class 0', 'Class 1'],
            yticklabels=metrics.class_names if hasattr(metrics, 'class_names') else ['Class 0', 'Class 1']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(output_path, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"📈 Saved confusion matrix to {cm_path}")

    def _prepare_epoch_data(self):
        """Extract epoch-level metrics for both train and validation"""
        train_metrics = []
        valid_metrics = []
        epochs = sorted(
            [int(k.split('_')[1]) for k in self.history['train'] if k.startswith('epoch_')],
            key=lambda x: x
        )

        for epoch in epochs:
            key = f'epoch_{epoch}'
            tdata = self.history['train'][key]
            losses = [b['ce_loss'] for b in tdata.get('losses', [])]
            avg_train_loss = sum(losses) / len(losses) if losses else None

            train_metrics.append({
                'epoch': epoch,
                'acc': tdata.get('acc'),
                'precision': tdata.get('precision'),
                'recall': tdata.get('recall'),
                'f1_macro': tdata.get('f1_macro'),
                'average_loss': avg_train_loss,
            })

            # Validation data
            vdata = self.history['valid'][key]
            valid_metrics.append({
                'epoch': epoch,
                'acc': vdata.get('acc'),
                'precision': vdata.get('precision'),
                'recall': vdata.get('recall'),
                'f1_macro': vdata.get('f1_macro'),
                'average_loss': vdata.get('average_loss'),
            })

        return pd.DataFrame(train_metrics), pd.DataFrame(valid_metrics)

    def _plot_loss(self, plt, train_df, valid_df):
        plt.plot(train_df['epoch'], train_df['average_loss'], marker='o',
                 linestyle='-', linewidth=2, label='Train CE Loss')
        plt.plot(valid_df['epoch'], valid_df['average_loss'], marker='o',
                 linestyle='--', linewidth=2, label='Validation CE Loss')
        plt.title(self.labels['loss'] + ' Evolution')
        plt.xlabel('Epoch')
        plt.ylabel(self.labels['loss'])
        plt.xticks(range(1, max(train_df['epoch']) + 1))
        plt.grid(True)
        plt.legend()

    def _plot_standard_metric(self, plt, metric, train_df, valid_df):
        plt.plot(train_df['epoch'], train_df[metric], marker='o', linestyle='-',
                 linewidth=2, label=f'Train {self.labels[metric]}')
        plt.plot(valid_df['epoch'], valid_df[metric], marker='o', linestyle='--',
                 linewidth=2, label=f'Validation {self.labels[metric]}')
        plt.title(f'{self.labels[metric]} Evolution')
        plt.xlabel('Epoch')
        plt.ylabel(self.labels[metric])
        plt.xticks(range(1, max(valid_df['epoch']) + 1))
        plt.grid(True)
        plt.legend()
