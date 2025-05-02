import os
import glob
import time
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
import psutil
from typing import Any, Dict

# -----------------------------------------------------------------------------
#  Hardware‑aware helpers
# -----------------------------------------------------------------------------

PROC = psutil.Process()
PROC.cpu_percent(interval=None)


def load_hardware_profile(profile_path: str) -> Dict[str, Any]:
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as ex:
            print(f"⚠️  Failed to read hardware profile {profile_path}: {ex}")
    return {}


def apply_hardware_constraints(profile: Dict[str, Any]):
    max_threads = profile.get("cpu_threads") or profile.get("cpu_cores")
    if isinstance(max_threads, int) and max_threads > 0:
        torch.set_num_threads(max_threads)
        os.environ["OMP_NUM_THREADS"] = str(max_threads)
        print(f"🔧  Thread cap applied: {max_threads}")


# -----------------------------------------------------------------------------
#  Public helpers – API stays unchanged
# -----------------------------------------------------------------------------

def find_best_model_path(output_path):
    paths = glob.glob(os.path.join(output_path, "best_*.pth"))
    return max(paths, key=os.path.getctime) if paths else None


def get_model_size(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)  # MB


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, input_shape=(1, 3, 224, 224)):
    from fvcore.nn import FlopCountAnalysis
    dummy = torch.randn(input_shape)
    return FlopCountAnalysis(model, dummy).total() / 1e9  # GFLOPs


def convert_to_onnx(model, example_input, onnx_path):
    torch.onnx.export(
        model.cpu(),
        example_input.cpu(),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return onnx_path


def benchmark_inference(session, input_data, duration_s: int = 10):
    latencies, cpu_utils, mem_usages = [], [], []
    start = time.time()
    while time.time() - start < duration_s:
        cpu_utils.append(PROC.cpu_percent(interval=None) / psutil.cpu_count())
        mem_usages.append(psutil.virtual_memory().used / (1024 * 1024))
        t0 = time.time()
        session.run(None, {"input": input_data})
        latencies.append((time.time() - t0) * 1000)
    arr = np.array(latencies)
    return {
        "runs": len(arr),
        "throughput_rps": len(arr) / duration_s,
        "latency_ms": arr.mean(),
        "latency_p90_ms": np.percentile(arr, 90),
        "latency_p99_ms": np.percentile(arr, 99),
        "cpu_utilization": np.mean(cpu_utils),
        "memory_usage_mb": np.mean(mem_usages),
    }

# -----------------------------------------------------------------------------
#  benchmark_onnx (signature unchanged)
# -----------------------------------------------------------------------------


def benchmark_onnx(model, data_iter, device, output_path, test_name):

    model = model.cpu().eval()
    hw_profile = load_hardware_profile("hardware_lmp.json")
    apply_hardware_constraints(hw_profile)
    sample_images, _ = next(iter(data_iter))
    sample_input = sample_images[0:1].cpu()
    onnx_path = os.path.join(output_path, f"{test_name}_model.onnx")
    convert_to_onnx(model, sample_input, onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(output_path, f"{test_name}_ort_profile")

    session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    results = benchmark_inference(session, sample_input.numpy(), duration_s=10)
    bench = {
        "model_size_mb": get_model_size(onnx_path),
        "num_parameters": count_model_parameters(model),
        "gflops": measure_flops(model, input_shape=sample_input.shape),
        **results,
        "hardware_profile": hw_profile or "default_host",
    }
    out_json = os.path.join(output_path, f"{test_name}_onnx_benchmark.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(bench, fh, indent=2)
    print("\n🚀  ONNX Runtime Benchmark Results (CPU‑only / 10‑sec window)")
    for k, v in bench.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")
    return bench
