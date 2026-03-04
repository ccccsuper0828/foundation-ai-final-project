"""
Efficiency benchmarks: FLOPs, parameter count, inference latency, peak memory.
"""

import time
import torch
import torch.nn as nn
from models.build_model import count_parameters


def measure_flops(model, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Measure FLOPs using fvcore (if available) or return estimate.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        model = model.to(device)
        model.eval()
        dummy = torch.randn(*input_size, device=device)
        flops = FlopCountAnalysis(model, dummy)
        return flops.total()
    except Exception as e:
        print(f"  [FLOPs] fvcore failed ({e}), returning -1")
        return -1


def measure_inference_latency(model, input_size=(1, 3, 224, 224), device="cpu", num_runs=100, warmup=10):
    """
    Measure average inference latency in milliseconds.
    """
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / num_runs * 1000
    return avg_ms


def measure_peak_memory(model, input_size=(1, 3, 224, 224)):
    """
    Measure peak GPU memory usage during forward + backward pass.
    Returns memory in MB. Only works on CUDA.
    """
    if not torch.cuda.is_available():
        return -1

    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    dummy = torch.randn(*input_size, device=device)
    output = model(dummy)
    loss = output.sum()
    loss.backward()

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    model.eval()
    return peak_mem


def benchmark_model(model, model_name: str, device="cpu"):
    """Run all efficiency benchmarks on a single model."""
    n_params = count_parameters(model)
    flops = measure_flops(model, device=device)
    latency = measure_inference_latency(model, device=device)
    peak_mem = measure_peak_memory(model) if torch.cuda.is_available() else -1

    results = {
        "model_name": model_name,
        "params": n_params,
        "params_M": round(n_params / 1e6, 2),
        "flops": flops,
        "flops_G": round(flops / 1e9, 2) if flops > 0 else -1,
        "latency_ms": round(latency, 2),
        "peak_memory_MB": round(peak_mem, 2),
    }

    print(f"  {model_name}: {results['params_M']}M params | "
          f"{results['flops_G']}G FLOPs | "
          f"{results['latency_ms']}ms/img | "
          f"{results['peak_memory_MB']}MB peak")

    return results
