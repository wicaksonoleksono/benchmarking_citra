# Benchmarking PyTorch Mobile Image Classifier

## Prerequisites

- Conda (Anaconda atau Miniconda) terpasang dan tersedia di PATH.
- Python 3.8+.

## 1. Membuat Environment Conda

```bash
# Buat environment baru dengan Python 3.9 (atau versi lain yang diinginkan)
conda create -n bench-imgcls python=3.9 -y
# Aktifkan environment
conda activate bench-imgcls
```

## 2. Install PyTorch dan Dependensi Utama

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install timm \
            numpy onnx onnxruntime \
            psutil Pillow \
            scikit-learn transformers \
            kagglehub pynvml matplotlib pandas ipykernell
```

atau bisa mengunjungi [Link ini](https://pytorch.org/get-started/locally) untuk instalasi pytorch

## 3. Struktur Direktori

```
├── pytorch
│   ├── benchmarker.py      # Skrip benchmarking
│   ├── config.py           # Konfigurasi eksperimen
│   ├── get_dataloader.py   # Loader dataset
│   ├── hardware_lmp.json   # Template hardware metrics
│   ├── main.py             # Entry point untuk train & eval
│   ├── model
│   │   └── MobileV3.py     # Definisi model MobileNetV3
│   ├── readme.md           # (Anda sedang melihatnya)
│   ├── tes.ipynb           # Notebook percobaan
│   ├── train_eval.py       # Fungsi train & evaluation
│   └── util.py             # Utility umum
```

## 4. Hardware Constraint (`hardware_lmp.json`)

File ini mendefinisikan batasan perangkat target untuk benchmarking: (nnti di standarisasi setelah LMP sdudah ada)

```json
{
  "cpu_cores": 2,
  "cpu_frequency_ghz": 1.1,
  "ram_gb": 4,
  "max_power_w": 50
}
```

Letakkan di `pytorch/hardware_lmp.json` agar skrip benchmarking dapat membaca dan menyesuaikan profil inferensi.

## 5. Konfigurasi Eksperimen (`config.py`)

Contoh `pytorch/config.py` menggunakan [EasyDict](https://github.com/makinacorpus/easydict):

```python
from easydict import EasyDict
import torch

cfg = EasyDict({
    # Device dan training
    "device": "cuda",  # atau "cpu"
    "num_epochs": 1,
    "data_params": {
        "batch_size": 32,
        "num_workers": 1,
    },
    "model": {
        "num_heads": 7,
    },
    "optimizer_class": torch.optim.SGD,
    "optimizer_name": "SGD",
    "optimizer_params": {
        "lr": 1e-2,
        "momentum": 0.9,
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
        "mobilenetv3_large_100.ra_in1k"
    ]
})
```

Skrip `main.py` akan menjalankan pipeline training & evaluation **secara paralel** untuk setiap entry di `model_names`.

## 6. Menjalankan Pipeline

Pastikan Anda berada di dalam folder `pytorch/`, lalu jalankan:

```bash
python main.py --configs "config.py" --workers 4
```

- `--configs`: path ke file konfigurasi.
- `--workers`: jumlah proses DataLoader.

---

Dengan README ini, Anda dapat langsung membuat environment, menginstall dependensi, memahami struktur, menetapkan batasan hardware, mengatur eksperimen, dan menjalankan benchmarking. Selamat mencoba!
