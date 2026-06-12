"""Command-line entry point for the Cat vs Dog semantic communication project.

Examples:
    python run_project.py prepare-captions --splits train test --max-samples none
    python run_project.py train --epochs 100 --batch-size 128
    python run_project.py evaluate --checkpoint checkpoints/multi_user_sc_catdog.pt
"""

import argparse
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)


def optional_int(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "all", "full"}:
        return None
    return int(text)


def prepare_captions(args):
    import prepare_catdog_captions as data_pipeline

    max_samples = optional_int(args.max_samples)
    for split in args.splits:
        split_max_samples = max_samples if split == "test" else None if args.full_train else max_samples
        data_pipeline.run_captioning(split=split, batch_size=args.caption_batch_size, max_samples=split_max_samples)

        if args.with_reconstruction:
            data_pipeline.run_reconstruction(
                split=split,
                batch_size=args.reconstruction_batch_size,
                max_samples=split_max_samples,
            )

        if args.with_clip_eval:
            data_pipeline.run_evaluation(split=split, max_samples=split_max_samples)


def train_model(args):
    import train_multi_user_semantic_comm as sc_system

    sc_system.set_seed(args.seed)
    sc_system.train_and_validate(
        train_json=args.train_json,
        val_json=args.val_json,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_users=args.num_users,
        use_channel_coding=not args.disable_channel_coding,
        use_semantic_encoder=not args.disable_semantic_encoder,
        d_model_sem=args.d_model_sem,
        dim_feedforward=args.dim_feedforward,
        dim_expand=args.dim_expand,
        dim_transmit=args.dim_transmit,
        checkpoint_path=args.checkpoint,
    )


def evaluate_model(args):
    import train_multi_user_semantic_comm as sc_system

    sc_system.evaluate_model_at_snrs(
        val_json=args.val_json,
        model_path=args.checkpoint,
        batch_size=args.batch_size,
        num_users=args.num_users,
        use_channel_coding=not args.disable_channel_coding,
        use_semantic_encoder=not args.disable_semantic_encoder,
        d_model_sem=args.d_model_sem,
        dim_feedforward=args.dim_feedforward,
        dim_expand=args.dim_expand,
        dim_transmit=args.dim_transmit,
    )

def evaluate_end_to_end(args):
    import train_multi_user_semantic_comm as sc_system
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load model
    best_model = sc_system.MultiUserSCSystem(
        vocab_size=tokenizer.vocab_size,
        num_users=args.num_users,
        d_model_sem=args.d_model_sem,
        dim_feedforward=args.dim_feedforward,
        dim_expand=args.dim_expand,
        dim_transmit=args.dim_transmit,
        use_channel_coding=not args.disable_channel_coding,
        use_semantic_encoder=not args.disable_semantic_encoder,
        max_len=25
    ).to(device)

    best_model.load_state_dict(torch.load(args.checkpoint, map_location=device))


    snrs = [int(x) for x in args.snrs]
    sc_system.run_end_to_end_pipeline(sc_model=best_model, test_snrs=snrs, max_samples=args.max_samples)

def build_parser():
    # 1. Create a parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--log-file", type=str, default="./outputs/training_log.txt", help="Path to save the log file")

    # 2. Main parser
    parser = argparse.ArgumentParser(description="Run the Cat vs Dog semantic communication project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 3. Add `parents=[parent_parser]` to EVERY subparser
    prep = subparsers.add_parser("prepare-captions", parents=[parent_parser], help="Generate BLIP caption JSON files and optional SD/CLIP outputs.")
    prep.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    prep.add_argument("--max-samples", default="none", help="Use an integer for a quick test, or 'none' for all samples.")
    prep.add_argument("--full-train", action="store_true", help="Always use the full train split even when max-samples is set.")
    prep.add_argument("--caption-batch-size", type=int, default=64)
    prep.add_argument("--reconstruction-batch-size", type=int, default=16)
    prep.add_argument("--with-reconstruction", action="store_true")
    prep.add_argument("--with-clip-eval", action="store_true")
    prep.set_defaults(func=prepare_captions)

    train = subparsers.add_parser("train", parents=[parent_parser], help="Train the MultiUserSCSystem model.")
    train.add_argument("--train-json", default="catdog_captions_train.json")
    train.add_argument("--val-json", default="catdog_captions_test.json")
    train.add_argument("--checkpoint", default="checkpoints/multi_user_sc_catdog.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--num-users", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--d-model-sem", type=int, default=128)
    train.add_argument("--dim-feedforward", type=int, default=256)
    train.add_argument("--dim-expand", type=int, default=512)
    train.add_argument("--dim-transmit", type=int, default=128)
    train.add_argument("--disable-channel-coding", action="store_true")
    train.add_argument("--disable-semantic-encoder", action="store_true")
    train.set_defaults(func=train_model)

    eval_parser = subparsers.add_parser("evaluate", parents=[parent_parser], help="Evaluate a trained checkpoint over SNR values.")
    eval_parser.add_argument("--val-json", default="catdog_captions_test.json")
    eval_parser.add_argument("--checkpoint", default="checkpoints/multi_user_sc_catdog.pt")
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--num-users", type=int, default=4)
    eval_parser.add_argument("--d-model-sem", type=int, default=128)
    eval_parser.add_argument("--dim-feedforward", type=int, default=256)
    eval_parser.add_argument("--dim-expand", type=int, default=512)
    eval_parser.add_argument("--dim-transmit", type=int, default=128)
    eval_parser.add_argument("--disable-channel-coding", action="store_true")
    eval_parser.add_argument("--disable-semantic-encoder", action="store_true")
    eval_parser.set_defaults(func=evaluate_model)

    e2e = subparsers.add_parser("evaluate-e2e", parents=[parent_parser], help="Run End-to-End Pipeline to evaluate SSQ and Images.")
    e2e.add_argument("--checkpoint", default="checkpoints/multi_user_sc_catdog.pt")
    e2e.add_argument("--num-users", type=int, default=4)
    e2e.add_argument("--snrs", nargs="+", default=[0, 2, 4, 6, 8, 10], help="List of SNR values to test")
    e2e.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation samples")
    e2e.add_argument("--d-model-sem", type=int, default=128)
    e2e.add_argument("--dim-feedforward", type=int, default=256)
    e2e.add_argument("--dim-expand", type=int, default=512)
    e2e.add_argument("--dim-transmit", type=int, default=128)
    e2e.add_argument("--disable-channel-coding", action="store_true")
    e2e.add_argument("--disable-semantic-encoder", action="store_true")
    e2e.set_defaults(func=evaluate_end_to_end)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    # --- DYNAMIC LOGGER SETUP ---
    import train_multi_user_semantic_comm as sc_system
    import sys
    import os

    # 1. Get the dynamic path from the CLI args
    log_path = args.log_file

    # 2. Extract the directory path and create it if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir:  # Ensure it doesn't fail if the path is just a filename
        os.makedirs(log_dir, exist_ok=True)

    # 3. Initialize the DualLogger
    sys.stdout = sc_system.DualLogger(log_path)
    print(f"[*] Logging output to: {log_path}")
    # ----------------------------
    args.func(args)



if __name__ == "__main__":
    main()
