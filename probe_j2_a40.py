import os
import subprocess
import sys


def run(command):
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    print(f"exit={completed.returncode}", flush=True)
    return completed.returncode


print("Python:", sys.version)
for name in (
    "HOSTNAME",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_GPUS",
    "CUDA_VISIBLE_DEVICES",
):
    print(f"{name}={os.environ.get(name)}")

run(["hostname"])
run(["nvidia-smi", "-L"])
run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
)

try:
    import torch
except Exception as exc:
    print(f"\nPyTorch import failed: {exc}")
else:
    print("\nPyTorch CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA device count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(
            f"torch device {idx}: {props.name}, "
            f"{props.total_memory / 1024**3:.2f} GiB"
        )
