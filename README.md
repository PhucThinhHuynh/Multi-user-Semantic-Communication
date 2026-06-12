# Cat vs Dog Multi-User Semantic Communication

This project prepares Cat vs Dog image captions with BLIP and trains a multi-user semantic communication model that transmits caption semantics through a noisy Rayleigh channel.

## Project Files

- `prepare_catdog_captions.py` prepares the dataset. It downloads `microsoft/cats_vs_dogs`, generates BLIP captions, optionally reconstructs images with Stable Diffusion, and optionally evaluates semantic similarity with CLIP.
- `train_multi_user_semantic_comm.py` contains the proposed `MultiUserSCSystem` model, training loop, SNR evaluation, and end-to-end visualization/evaluation utilities.
- `run_project.py` is the recommended command-line runner for reproducible execution.
- `requirements.txt` lists the required Python packages.

## Environment Setup

Python 3.8 is recommended. A CUDA-capable GPU is strongly recommended because BLIP, Stable Diffusion, CLIP, and model training are computationally heavy.

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your machine needs a specific CUDA build of PyTorch, install PyTorch first using the command from https://pytorch.org/get-started/locally/, then run:

```bash
pip install -r requirements.txt
```

## Step 1: Generate Caption JSON Files

For full training and test caption files:

```bash
python run_project.py prepare-captions --splits train test --max-samples none

```

This creates:

- `catdog_captions_train.json`
- `catdog_captions_test.json`

For a quick smoke test on 10 test images:

```bash
python run_project.py prepare-captions --splits test --max-samples 10
```

To also run Stable Diffusion reconstruction and CLIP evaluation:

```bash
python run_project.py prepare-captions --splits test --max-samples 10 --with-reconstruction --with-clip-eval --log-file ./outputs/test.txt
```

## Step 2: Train the Semantic Communication Model

After generating `catdog_captions_train.json` and `catdog_captions_test.json`, run:

```bash
python run_project.py train --epochs 100 --batch-size 128 --num-users 4  --log-file ./outputs/training_log.txt
```

The trained checkpoint is saved to:

```text
checkpoints/multi_user_sc_catdog.pt
```

## Step 3: Evaluate a Trained Checkpoint
To evaluate BLEU-1, BLEU-2 for system, run:
```bash
python run_project.py evaluate --checkpoint checkpoints/multi_user_sc_catdog.pt --batch-size 128 --num-users 4 --log-file ./outputs/Eval_BLEU.txt
```
To evaluate SSQ for system, run:
```bash
python run_project.py evaluate-e2e --checkpoint checkpoints/multi_user_sc_catdog.pt --num-users 4 --log-file ./outputs/eval_SSQ.txt
```

The evaluator reports BLEU-1, BLEU-2, SSQ, and related SNR-based metrics.

## Running the Original Scripts Directly

You can still run the two main scripts directly:

```bash
python prepare_catdog_captions.py
python train_multi_user_semantic_comm.py
```

However, `run_project.py` is preferred because it exposes arguments without editing source code.

If you want to train the version without EMC, toggle the use_emc flag in the class `MultiUserSCSystem` class of `train_multi_user_semantic_comm.py`.
## Application Focus and Modifications

The submission focuses on multi-user transmission using cross-modal and semantic embedding multiplexing, rather than serving as a direct clone of an existing repository. The main custom components are:

- A BLIP-based caption preparation pipeline for Cat vs Dog images.
- A Transformer-based multi-user semantic communication system.
- A trainable user projection matrix that acts as the proposed EMC Encoder user mask/extension module.
- EMC Encoder compression before channel coding.
- EMC Decoder extension, user masking, and compression before caption decoding.
- SNR-based validation with BLEU and SSQ metrics.

Generated files such as checkpoints, reconstructed images, and evaluation samples are ignored by `.gitignore` so the repository stays focused on source code and reproducible instructions.
