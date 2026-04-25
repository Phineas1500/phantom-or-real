# Scholar Cluster Command Reference

This is a practical command sheet for using Purdue Scholar GPUs for this project.

## Connect

```bash
ssh skiron@scholar.rcac.purdue.edu
```

Scholar login uses BoilerKey-style authentication. The password format is usually:

```text
PIN,DUO_PASSKEY
```

or, if enabled:

```text
PIN,push
```

## Project Locations

Home/project copy:

```bash
cd ~/phantom-or-real
```

Scratch/project copy:

```bash
cd /scratch/scholar/$USER/phantom-or-real
```

For this account:

```bash
cd /scratch/scholar/skiron/phantom-or-real
```

## Check Cluster Resources

Show your available queues/accounts:

```bash
slist
```

Show node features and GPU types:

```bash
sfeatures
```

Show node state, GPU resources, and features:

```bash
sinfo -N -o "%N %t %G %f"
```

Useful GPU node classes observed on Scholar:

```text
G nodes: 1x NVIDIA V100, 32 GB class
H nodes: 2x NVIDIA A30, 24 GB each
H-MIG/I-MIG nodes: A30 MIG slices, about 6 GB each
J nodes: 2x NVIDIA A40, about 46,068 MiB each
```

For Gemma 3 27B unquantized/BF16 inference, prefer J nodes:

```text
J node total GPU memory: 2x A40 = about 92,136 MiB total
Single A40 memory: about 46,068 MiB
```

The GPUs are separate devices. You only get the combined memory if your code shards the model across both GPUs, for example with Hugging Face `device_map="auto"`.

## Conda Environment

Load conda:

```bash
module load conda/2024.09
```

List environments:

```bash
conda env list
```

Create the project `phantom` environment from the pinned lockfile.

The CUDA, PyTorch, and vLLM wheels are large, so keep the environment, conda
package cache, pip cache, and temp files on scratch instead of the default home
env directory:

```bash
cd /scratch/scholar/$USER/phantom-or-real

export PHANTOM_ENV=/scratch/scholar/$USER/conda-envs/phantom
export CONDA_PKGS_DIRS=/scratch/scholar/$USER/conda-pkgs
export PIP_CACHE_DIR=/scratch/scholar/$USER/pip-cache
export TMPDIR=/scratch/scholar/$USER/tmp

mkdir -p /scratch/scholar/$USER/conda-envs "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
conda env create --prefix "$PHANTOM_ENV" -f environment.lock.yml
conda activate "$PHANTOM_ENV"
```

For this account, the recreated environment is here:

```text
/scratch/scholar/skiron/conda-envs/phantom
```

Use the looser portable spec only when you want conda/pip to resolve fresh
compatible versions:

```bash
conda env create --prefix "$PHANTOM_ENV" -f environment.yml
```

Activate the recreated environment in new shells:

```bash
module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom
```

Quick verification:

```bash
python -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)"
```

Optional Jupyter registration, only if notebooks are needed:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name phantom --display-name "Python [phantom]"
```

## Submit Jobs

Make a log directory:

```bash
mkdir -p slurm_logs
```

Submit a batch job:

```bash
sbatch job.sbatch
```

Submit and override selected SBATCH values from the command line:

```bash
sbatch --time=00:10:00 --mem=16G job.sbatch
```

## 2x A40 J-Node Probe

This request was tested successfully from this project:

```bash
sbatch probe_j2_a40.sbatch
```

The tested allocation completed on `scholar-j001` with:

```text
--constraint=J
--gres=gpu:2
CUDA_VISIBLE_DEVICES=0,1
GPU 0: NVIDIA A40, 46068 MiB total
GPU 1: NVIDIA A40, 46068 MiB total
```

## Recommended Gemma 3 27B Slurm Header

Use this shape for a two-A40 inference job:

```bash
#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --constraint=J
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --job-name=gemma3_27b
#SBATCH --output=slurm_logs/gemma3_27b_%j.out
#SBATCH --error=slurm_logs/gemma3_27b_%j.err

set -euo pipefail

module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python3 run_gemma.py
```

Important: `--mem=100G` reserves CPU RAM, not GPU VRAM. GPU VRAM comes from the GPUs reserved by `--gres=gpu:2`.

## Hugging Face Multi-GPU Load Pattern

Use `device_map="auto"` so the model can be placed across both A40s:

```python
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(model_id)
```

## Monitor Jobs

Show your jobs:

```bash
squeue -u $USER
```

Show one job:

```bash
squeue -j JOB_ID
```

Show selected fields:

```bash
squeue -j JOB_ID -o "%i %T %M %D %R %b"
```

Detailed job info:

```bash
scontrol show job JOB_ID
```

Watch output:

```bash
tail -f slurm_logs/JOB_OUTPUT_FILE.out
```

Cancel one job:

```bash
scancel JOB_ID
```

Cancel all of your jobs:

```bash
scancel -u $USER
```

Cancel only pending jobs:

```bash
scancel -t PENDING -u $USER
```

## Inspect Completed Jobs

Show accounting summary:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES%80
```

Show resource efficiency after completion:

```bash
seff JOB_ID
```

Show detailed accounting:

```bash
sacct -j JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
```

## Monitor Running Job Resources

```bash
sstat -j JOB_ID --format=AveCPU,AvePages,AveRSS,AveVMSize
```

To inspect GPUs from inside the job:

```bash
nvidia-smi
```

Compact GPU memory report:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
```

## Interactive Sessions

Interactive CPU session:

```bash
sinteractive -A scholar --time=2:00:00
```

Interactive GPU session:

```bash
sinteractive -A gpu --gres=gpu:1 --time=2:00:00
```

Interactive J-node A40 session:

```bash
sinteractive -A gpu --constraint=J --gres=gpu:2 --time=2:00:00
```

Quick debug session:

```bash
sinteractive -A debug --time=15:00
```

## File Transfer

Copy a file to Scholar:

```bash
scp local_file.py skiron@scholar.rcac.purdue.edu:~/
```

Copy a directory to Scholar:

```bash
scp -r local_directory/ skiron@scholar.rcac.purdue.edu:~/
```

Copy a file from Scholar to local:

```bash
scp skiron@scholar.rcac.purdue.edu:~/remote_file.py ./
```

## Disk Usage and Cleanup

Check large files/directories:

```bash
du -sh ~/*
du -sh /scratch/scholar/$USER/*
```

Clean conda cache:

```bash
conda clean --all
```

Check loaded modules:

```bash
module list
```

List available modules:

```bash
module avail
```

## Common Issues

If `conda` is not found:

```bash
module load conda/2024.09
```

If the job script is not executable:

```bash
chmod +x job.sbatch
```

If a job exits immediately, check both logs:

```bash
cat slurm_logs/JOB_NAME_JOBID.out
cat slurm_logs/JOB_NAME_JOBID.err
```

If CUDA is not visible inside a job:

```bash
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Rules of Thumb

Do not run heavy training or inference directly on frontend nodes.

Use `debug` for short tests, `gpu` for GPU jobs, and `J` constraint for A40 jobs.

For Gemma 3 27B unquantized inference, request both A40s on a J node and use model sharding.

For full fine-tuning Gemma 3 27B, 2x A40 is likely too small unless you use LoRA/QLoRA, offload, checkpointing, and very small batches.
