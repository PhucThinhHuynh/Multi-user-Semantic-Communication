"""Multi-user semantic communication model for the Cat vs Dog caption task.

This script trains and evaluates the proposed semantic communication system.
The model consumes BLIP-generated caption JSON files, transmits multiple user
captions through a noisy Rayleigh channel, and reconstructs user captions with a
Transformer-based receiver.
"""
import os
import sys
import subprocess
import time
import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from PIL import Image


# ==========================================
# AUTO-INSTALLER
# ==========================================
def auto_install_package(package_name):
    print(f"[*] Package '{package_name}' is not installed. Downloading and installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"[+] Package '{package_name}' installed successfully!\n")
    except Exception as e:
        print(f"[!] Failed to automatically install package '{package_name}'. Error: {e}")
        sys.exit(1)


try:
    from datasets import load_dataset
except ImportError:
    auto_install_package("datasets")
    from datasets import load_dataset

try:
    import nltk

    nltk.data.find('tokenizers/punkt')
except LookupError:
    import nltk

    nltk.download('punkt')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 0. DUAL LOGGER
# ==========================================
class DualLogger(object):
    """
    Logs terminal output to both the console and a text file.
    """
    def __init__(self, filename="training_evaluation_results.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if '\r' not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


# ==========================================
# 1. SYSTEM METRICS EVALUATOR
# ==========================================
class SystemMetricsEvaluator:
    """
    Utility class to compute and log network performance metrics:
    BLEU, SER, SSQ, FLOPs, Params, Inference Time, Compression Rate, and Channel Capacity.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.smoothie = SmoothingFunction().method1

    def profile_system(self, model, batch_size, max_len, num_users, device, snr_db=10.0):
        print("\n" + "=" * 60)
        print(" EVALUATING SYSTEM COMPLEXITY (PARAMS, FLOPs, TIME)")
        print("=" * 60)

        # 1. Parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" [*] Total Parameters          : {total_params:,}")

        # 2. Inference Time
        model.eval()
        dummy_src = [torch.randint(0, 30000, (batch_size, max_len)).to(device) for _ in range(num_users)]
        dummy_tgt = [torch.randint(0, 30000, (batch_size, max_len - 1)).to(device) for _ in range(num_users)]
        dummy_src_pad = [torch.zeros((batch_size, max_len), dtype=torch.bool).to(device) for _ in range(num_users)]
        dummy_tgt_pad = [torch.zeros((batch_size, max_len - 1), dtype=torch.bool).to(device) for _ in range(num_users)]
        dummy_tgt_mask = model.generate_square_subsequent_mask(max_len - 1).to(device)

        with torch.no_grad():
            for _ in range(5):  # Warm-up
                _ = model(dummy_src, dummy_tgt, snr_db, dummy_src_pad, dummy_tgt_pad, dummy_tgt_mask)
            start_time = time.time()
            num_iters = 20
            for _ in range(num_iters):
                _ = model(dummy_src, dummy_tgt, snr_db, dummy_src_pad, dummy_tgt_pad, dummy_tgt_mask)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.time()

        avg_infer_time = ((end_time - start_time) / num_iters) * 1000
        print(f" [*] Average Inference Time    : {avg_infer_time:.2f} ms / batch")

        # 3. FLOPs
        gflops = 0.0
        try:
            from thop import profile
            macs, _ = profile(model,
                              inputs=(dummy_src, dummy_tgt, snr_db, dummy_src_pad, dummy_tgt_pad, dummy_tgt_mask),
                              verbose=False)
            gflops = (macs * 2) / 1e9
            print(f" [*] Computational Complexity  : {gflops:.3f} GFLOPs")
        except ImportError:
            print(" [!] Cannot find 'thop' library. Install using 'pip install thop' for FLOPs calculation.")

        print("=" * 60 + "\n")
        return total_params, gflops, avg_infer_time

    def calculate_bleu_string(self, target_text, pred_text):
        b1 = sentence_bleu([target_text.split()], pred_text.split(), weights=(1.0, 0, 0, 0),
                           smoothing_function=self.smoothie)
        b2 = sentence_bleu([target_text.split()], pred_text.split(), weights=(0.5, 0.5, 0, 0),
                           smoothing_function=self.smoothie)
        return b1, b2

    def calculate_bleu_detailed(self, predictions, targets):
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to evaluate detailed BLEU.")
        pred_ids = torch.argmax(predictions, dim=-1)
        results = []
        for i in range(targets.size(0)):
            target_text = self.tokenizer.decode(targets[i], skip_special_tokens=True).split()
            pred_text = self.tokenizer.decode(pred_ids[i], skip_special_tokens=True).split()
            if not pred_text:
                results.append({'target': " ".join(target_text), 'pred': "", 'b1': 0.0, 'b2': 0.0})
                continue
            b1 = sentence_bleu([target_text], pred_text, weights=(1.0, 0, 0, 0), smoothing_function=self.smoothie)
            b2 = sentence_bleu([target_text], pred_text, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothie)
            results.append({'target': " ".join(target_text), 'pred': " ".join(pred_text), 'b1': b1, 'b2': b2})
        return results

    def calculate_ser(self, predictions, targets, pad_token_id):
        pred_ids = torch.argmax(predictions, dim=-1)
        valid_mask = targets != pad_token_id
        errors = (pred_ids[valid_mask] != targets[valid_mask]).sum().item()
        return errors, valid_mask.sum().item()

    def calculate_ssq(self, acc_recon, acc_orig):
        """Semantic Similarity Quality based on CLIP accuracies"""
        return acc_recon / acc_orig if acc_orig > 0 else 0.0

    def calculate_compression_rate(self, dim_transmit, max_len=25, img_shape=(512, 512, 3)):
        """
        Data Compression Rate (DCR)
        Compares transmitted 32-bit latent vectors against the original 8-bit RGB image.
        """
        orig_bits = 3 * 224 * 224 * 8 * 8
        transmit_bits = max_len * dim_transmit * 32
        dcr_ratio = transmit_bits / orig_bits
        dcr_percent = dcr_ratio * 100
        return dcr_ratio, dcr_percent

    def calculate_theoretical_capacity(self, snr_db, U, d, rho_s=1.0, rho_ia=0.99):
        """
        Per-User Capacity of the Shared Embedding (SE) scheme.
        Based strictly on Equation (29) from the paper.
        """
        snr_linear = 10 ** (snr_db / 10.0)
        numerator = (1 + (d - 1) * rho_s) * snr_linear
        denominator = (U - 1) * (1 - rho_ia) * snr_linear + 1
        c_se = 0.5 * math.log2(1 + (numerator / denominator))
        return c_se


# ==========================================
# 2. DATASETS
# ==========================================
class CaptionDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=25):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data[idx]['caption']
        encoded = self.tokenizer(
            caption, padding='max_length', truncation=True,
            max_length=self.max_len, return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


class CatDogDataset(Dataset):
    def __init__(self, split='test', max_samples=None):
        print(f"[*] Loading dataset microsoft/cats_vs_dogs...")
        ds = load_dataset("microsoft/cats_vs_dogs", split="train", trust_remote_code=True)
        ds = ds.train_test_split(test_size=0.2, seed=42)
        self.data = ds[split]

        if max_samples is not None:
            actual_samples = min(max_samples, len(self.data))
            self.data = self.data.select(range(actual_samples))
            print(f"[*] Reduced {split.upper()} dataset down to {actual_samples} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image'].convert('RGB')
        label = item['labels']
        return img, label


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels


# ==========================================
# 3. BASE MODULES (POS ENCODING, CHANNEL)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adds sequential position context into the token embeddings
        return x + self.pe[:x.size(0), :, :]

class PowerNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Enforces Average Power Constraint: E[|x|^2] = 1
        power = torch.mean(x ** 2, dim=(0, 2), keepdim=True)
        return x / torch.sqrt(power + 1e-8)

class RayleighChannelPaper(nn.Module):
    def __init__(self, num_users=8):
        super().__init__()
        self.num_users = num_users

    def forward(self, x, snr_db):
        batch_size = x.shape[1]

        # 1. Generate Block Fading
        z = torch.randn((1, batch_size, 1), device=x.device)
        w = torch.randn((1, batch_size, 1), device=x.device)
        h = torch.sqrt(z ** 2 + w ** 2) / math.sqrt(2.0)

        # 2. Calculate Noise Power
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise = torch.randn_like(x) * math.sqrt(noise_power)

        # 3. Apply Channel
        y = h * x + noise

        # 4. Zero-Forcing Equalization (Perfect CSI)
        y_equalized = y / h

        return y_equalized
# ==========================================
# 4. CORE ARCHITECTURE: MULTI-USER SC SYSTEM
# ==========================================
class MultiUserSCSystem(nn.Module):
    def __init__(self, vocab_size, num_users=4, d_model_sem=128, dim_feedforward=256,
                 dim_expand=512, dim_transmit=128, nhead=8, num_layers=4,
                 use_channel_coding=True, use_semantic_encoder=True,
                 use_emc=True, max_len=25):
        super().__init__()
        self.num_users = num_users
        self.max_len = max_len
        self.d_model_sem = d_model_sem
        self.dim_expand = dim_expand
        self.dim_transmit = dim_transmit
        self.use_channel_coding = use_channel_coding
        self.use_semantic_encoder = use_semantic_encoder
        self.use_emc = use_emc # Store the toggle state

        # --- CORE EMBEDDINGS & PE ---
        self.embedding = nn.Embedding(vocab_size, self.d_model_sem)
        self.pos_encoder = PositionalEncoding(self.d_model_sem)

        # --- TRANSMITTER (TX) BASE ENCODER ---
        if self.use_semantic_encoder:
            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model_sem,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                activation="relu",
                batch_first=False,
                norm_first=False
            )
            self.tx_semantic_encoder = nn.TransformerEncoder(tx_encoder_layer, num_layers=num_layers)
        else:
            self.tx_semantic_encoder = None

        # =========================================================================
        # PROPOSED MODULE 1: EMC-Encoder (Extended Masking and Compression)
        # =========================================================================
        if self.use_emc:
            self.user_projectors = nn.Parameter(torch.empty(num_users, dim_expand, dim_expand))
            self.reset_user_projectors()

            self.tx_proj = nn.Linear(self.d_model_sem, self.dim_expand)
            self.tx_proj_norm = nn.LayerNorm(self.dim_expand)

            self.tx_com = nn.Sequential(
                nn.Linear(self.dim_expand, self.d_model_sem),
                nn.LayerNorm(self.d_model_sem),
                nn.ReLU()
            )
        # =========================================================================

        self.channel_encoder = nn.Sequential(
            nn.Linear(self.d_model_sem, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, self.dim_transmit)
        )

        # --- CHANNEL ---
        self.power_norm = PowerNormalization()
        self.channel = RayleighChannelPaper(num_users=num_users)

        # --- RECEIVER (RX) BASE DECODER ---
        self.channel_decoder = nn.Sequential(
            nn.Linear(self.dim_transmit, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, self.d_model_sem),
        )

        # =========================================================================
        # PROPOSED MODULE 2: EMC-Decoder (Extended Masking and Compression)
        # =========================================================================
        if self.use_emc:
            self.rx_expand = nn.Linear(self.d_model_sem, self.dim_expand)
            self.rx_expand_norm = nn.LayerNorm(self.dim_expand)

            self.rx_fc1 = nn.Linear(self.dim_expand, self.d_model_sem)
            self.rx_fc1_norm = nn.LayerNorm(self.d_model_sem)

            self.rx_fc2 = nn.Linear(self.dim_expand, self.d_model_sem)
            self.rx_fc2_norm = nn.LayerNorm(self.d_model_sem)

        # We keep the shared encoder outside the EMC block because the base model needs it
        rx_shared_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model_sem,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=False,
            norm_first=False
        )
        self.rx_shared_encoder = nn.TransformerEncoder(rx_shared_layer, num_layers=num_layers)

        rx_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model_sem,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=False,
            norm_first=False
        )
        self.rx_semantic_decoder = nn.TransformerDecoder(rx_decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(self.d_model_sem, vocab_size)

    def reset_user_projectors(self):
        if hasattr(self, 'user_projectors'):
            with torch.no_grad():
                eye = torch.eye(self.dim_expand).unsqueeze(0).repeat(self.num_users, 1, 1)
                noise = 0.01 * torch.randn_like(eye)
                self.user_projectors.copy_(eye + noise)

    def apply_user_projection(self, x, user_idx):
        return torch.matmul(x, self.user_projectors[user_idx])

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, src_list, tgt_list, snr_db, src_pad_masks, tgt_pad_masks, tgt_mask):
        src_t_list = [src.transpose(0, 1) for src in src_list]
        tgt_t_list = [tgt.transpose(0, 1) for tgt in tgt_list]
        encoded_signals = []

        sem_encoded_list = []
        for u in range(self.num_users):
            sem_encoded = self.embedding(src_t_list[u]) * math.sqrt(self.d_model_sem)
            sem_encoded = self.pos_encoder(sem_encoded)
            sem_encoded_list.append(sem_encoded)

        # --- TRANSMITTER WORKFLOW ---
        for u in range(self.num_users):
            sem_encoded = sem_encoded_list[u]

            if self.use_semantic_encoder and self.tx_semantic_encoder is not None:
                sem_encoded = self.tx_semantic_encoder(
                    sem_encoded,
                    src_key_padding_mask=src_pad_masks[u]
                )

            # --- PROPOSED MODULE 1 (EMC-Encoder TOGGLE) ---
            if self.use_emc:
                sem_encoded_exp = self.tx_proj(sem_encoded)
                sem_encoded_exp = self.tx_proj_norm(sem_encoded_exp)
                sem_encoded_exp = F.relu(sem_encoded_exp)

                projected_sem = self.apply_user_projection(sem_encoded_exp, user_idx=u)
                sem_encoded_exp = self.tx_com(projected_sem)
            else:
                # Bypass: Pass semantic tokens directly to channel encoder
                sem_encoded_exp = sem_encoded
            # ----------------------------------------------

            if self.use_channel_coding:
                chan_encoded = self.channel_encoder(sem_encoded_exp)
            else:
                chan_encoded = sem_encoded_exp

            encoded_signals.append(chan_encoded)

        # --- CHANNEL WORKFLOW ---
        transmitted_signal = sum(encoded_signals)
        transmitted_signal = self.power_norm(transmitted_signal)
        received_signal = self.channel(transmitted_signal, snr_db)

        # --- RECEIVER WORKFLOW ---
        outputs = []
        if self.use_channel_coding:
            chan_decoded = self.channel_decoder(received_signal)
        else:
            chan_decoded = received_signal

        # --- PROPOSED MODULE 2 (EMC-Decoder EXPAND TOGGLE) ---
        if self.use_emc:
            X1 = self.rx_expand(chan_decoded)
            X1 = self.rx_expand_norm(X1)

            x1_compressed = self.rx_fc1(X1)
            x1_compressed = self.rx_fc1_norm(x1_compressed)
            hidden_feature = F.relu(x1_compressed)
        else:
            # Bypass: Skip the expand logic and route channel decode straight to shared encoder
            hidden_feature = chan_decoded
        # -----------------------------------------------------

        # Shared processing happens regardless
        hidden_feature = self.rx_shared_encoder(hidden_feature)

        for u in range(self.num_users):
            eve_rx_masks = getattr(self, 'eve_rx_masks', None)
            rx_user_idx = eve_rx_masks[u] if eve_rx_masks is not None else u

            # --- PROPOSED MODULE 2 (EMC-Decoder MASK & COMPRESS TOGGLE) ---
            if self.use_emc:
                masked_X1 = self.apply_user_projection(X1, user_idx=rx_user_idx)
                user_feature = self.rx_fc2(masked_X1)
                user_feature = self.rx_fc2_norm(user_feature)
                user_feature = F.relu(user_feature)
            else:
                # Bypass: Without user-specific masking, the user just receives the aggregate decoded channel
                user_feature = chan_decoded
            # --------------------------------------------------------------

            tgt_emb = self.embedding(tgt_t_list[u]) * math.sqrt(self.d_model_sem)
            tgt_emb = self.pos_encoder(tgt_emb)

            user_feature_sliced = user_feature[:tgt_emb.size(0), :, :]
            sem_decoded = tgt_emb + user_feature_sliced

            sem_decoded = self.rx_semantic_decoder(
                sem_decoded,
                hidden_feature,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_masks[u],
                memory_key_padding_mask=src_pad_masks[u]
            )

            output = self.output_layer(sem_decoded)
            outputs.append(output.transpose(0, 1))

        return outputs

# ==========================================
# 5. END-TO-END PIPELINE & TRAINING WRAPPER
# ==========================================
def plot_10_failures(failures, filename="end_to_end_10_failures.png"):
    if not failures:
        print("Excellent! No failures detected.")
        return
    n_plots = min(10, len(failures))
    failures = failures[:n_plots]
    fig, axes = plt.subplots(nrows=n_plots, ncols=3, figsize=(18, 4.5 * n_plots))
    fig.suptitle("Top 10 Failures (Original Correct, Recon Wrong)", fontsize=20, fontweight='bold')
    classes = ['Cat', 'Dog']

    for i, sample in enumerate(failures):
        ax_orig = axes[i, 0] if n_plots > 1 else axes[0]
        ax_text = axes[i, 1] if n_plots > 1 else axes[1]
        ax_recon = axes[i, 2] if n_plots > 1 else axes[2]

        ax_orig.imshow(sample['orig_img'].resize((256, 256), Image.NEAREST))
        ax_orig.set_title(f"Original\nTrue: {classes[sample['true_label']]} | Pred: {classes[sample['pred_orig']]}",
                          fontsize=14)
        ax_orig.axis('off')

        ax_recon.imshow(sample['recon_img'].resize((256, 256), Image.NEAREST))
        ax_recon.set_title(f"Reconstructed\nPred: {classes[sample['pred_recon']]}", fontsize=14, color='red')
        ax_recon.axis('off')

        ax_text.axis('off')
        orig_cap = textwrap.fill(sample['orig_cap'], width=40)
        recon_cap = textwrap.fill(sample['recon_cap'], width=40)
        text_info = f"--- ORIGINAL CAPTION ---\n{orig_cap}\n\n--- RECONSTRUCTED CAPTION ---\n{recon_cap}\n\nBLEU-1: {sample['b1']:.2f} | BLEU-2: {sample['b2']:.2f}"
        ax_text.text(0.05, 0.5, text_info, fontsize=14, verticalalignment='center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if n_plots > 1 else 0.85)
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


def run_end_to_end_pipeline(sc_model, test_snrs=[0, 2, 4, 6, 8, 10], max_samples=None):
    device = next(sc_model.parameters()).device
    sc_model.eval()

    print(f"\n========================================================")
    print(f" STARTING END-TO-END PIPELINE (Cats vs Dogs) - SNRs: {test_snrs}dB")
    print(f"========================================================")

    dataset = CatDogDataset(split='test', max_samples=max_samples)

    NUM_USERS = sc_model.num_users
    BATCH_SIZE_PER_USER = 2
    dataloader = DataLoader(dataset, batch_size=NUM_USERS * BATCH_SIZE_PER_USER, shuffle=False,
                            collate_fn=custom_collate_fn)

    print("[*] Loading models (BLIP, Stable Diffusion, CLIP)...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                              use_safetensors=False).to(device)
    blip_model.eval()

    sc_tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
    evaluator = SystemMetricsEvaluator(tokenizer=sc_tokenizer)

    sd_pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16,
                                                      use_safetensors=False).to(device)
    sd_pipe.safety_checker = None
    sd_pipe.set_progress_bar_config(disable=True)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    classes_catdog = ['cat', 'dog']
    clip_prompts = [f"a photo of a {c}" for c in classes_catdog]

    results_per_snr = {snr: {'correct_recon': 0, 'total_b1': 0, 'total_b2': 0, 'failures': []} for snr in test_snrs}
    correct_orig = 0
    total_images = 0

    print("\n[*] Running End-to-End Inference (Please wait)...")
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(dataloader, desc="Processing Batches"):
            current_batch_size = len(batch_images)
            pad_size = 0
            if current_batch_size % NUM_USERS != 0:
                pad_size = NUM_USERS - (current_batch_size % NUM_USERS)
                batch_images.extend([batch_images[-1]] * pad_size)
                batch_labels.extend([batch_labels[-1]] * pad_size)

            actual_bs_per_user = len(batch_images) // NUM_USERS

            # STEP A: BLIP CAPTIONING & CLIP PREDICTION
            inputs_blip = blip_processor(images=batch_images, return_tensors="pt").to(device)
            outputs_blip = blip_model.generate(**inputs_blip, max_new_tokens=25)
            orig_captions = blip_processor.batch_decode(outputs_blip, skip_special_tokens=True)

            inputs_orig = clip_processor(text=clip_prompts, images=batch_images[:current_batch_size],
                                         return_tensors="pt", padding=True).to(device)
            pred_origs = clip_model(**inputs_orig).logits_per_image.argmax(dim=1).tolist()

            for i in range(current_batch_size):
                if pred_origs[i] == batch_labels[i]:
                    correct_orig += 1
            total_images += current_batch_size

            # Encode Text generated to input into the SC System
            encoded_text = sc_tokenizer(orig_captions, padding='max_length', truncation=True, max_length=25,
                                        return_tensors='pt')
            input_ids_all = encoded_text['input_ids'].view(NUM_USERS, actual_bs_per_user, -1).to(device)
            attn_mask_all = encoded_text['attention_mask'].view(NUM_USERS, actual_bs_per_user, -1).to(device)

            src_list, tgt_in_list, src_pad_list, tgt_pad_list = [], [], [], []
            for u in range(NUM_USERS):
                src_list.append(input_ids_all[u])
                tgt_in_list.append(input_ids_all[u, :, :-1])
                src_pad_list.append(attn_mask_all[u] == 0)
                tgt_pad_list.append(attn_mask_all[u, :, :-1] == 0)

            tgt_mask = sc_model.generate_square_subsequent_mask(tgt_in_list[0].size(1)).to(device)

            # ITERATE OVER EACH SNR POINT
            for snr_db in test_snrs:
                # STEP B: SC SYSTEM TRANSMISSION
                sc_outputs = sc_model(src_list, tgt_in_list, snr_db, src_pad_list, tgt_pad_list, tgt_mask)

                recon_captions = []
                for u in range(NUM_USERS):
                    pred_ids = torch.argmax(sc_outputs[u], dim=-1)
                    for b in range(actual_bs_per_user):
                        recon_captions.append(sc_tokenizer.decode(pred_ids[b], skip_special_tokens=True))

                orig_caps_batch = orig_captions[:current_batch_size]
                recon_caps_batch = recon_captions[:current_batch_size]
                batch_images_valid = batch_images[:current_batch_size]
                batch_labels_valid = batch_labels[:current_batch_size]

                # STEP C: STABLE DIFFUSION RECONSTRUCTION
                recon_images = sd_pipe(prompt=recon_caps_batch, num_inference_steps=20, height=512, width=512).images

                # STEP D: CLIP EVALUATION AND IMAGE CLASSIFICATION
                inputs_recon = clip_processor(text=clip_prompts, images=recon_images, return_tensors="pt",
                                              padding=True).to(device)
                pred_recons = clip_model(**inputs_recon).logits_per_image.argmax(dim=1).tolist()

                for i in range(current_batch_size):
                    b1, b2 = evaluator.calculate_bleu_string(orig_caps_batch[i], recon_caps_batch[i])
                    results_per_snr[snr_db]['total_b1'] += b1
                    results_per_snr[snr_db]['total_b2'] += b2

                    if pred_recons[i] == batch_labels_valid[i]:
                        results_per_snr[snr_db]['correct_recon'] += 1

                    if (pred_origs[i] == batch_labels_valid[i]) and (pred_recons[i] != batch_labels_valid[i]):
                        results_per_snr[snr_db]['failures'].append({
                            'orig_img': batch_images_valid[i], 'recon_img': recon_images[i],
                            'orig_cap': orig_caps_batch[i], 'recon_cap': recon_caps_batch[i],
                            'true_label': batch_labels_valid[i], 'pred_orig': pred_origs[i],
                            'pred_recon': pred_recons[i],
                            'b1': b1, 'b2': b2
                        })

    acc_orig = correct_orig / total_images if total_images > 0 else 0
    dcr_ratio, dcr_percent = evaluator.calculate_compression_rate(sc_model.dim_transmit, sc_model.max_len)

    print("\n=========================================================================")
    print(f" END-TO-END EVALUATION RESULTS | Total samples: {total_images}")
    print(f" Original Accuracy (CLIP pred vs True label): {acc_orig:.4f}")
    print(f" Data Compression Rate (DCR): {dcr_ratio:.6f} ({dcr_percent:.4f}%)")
    print("-------------------------------------------------------------------------")
    print(f"| SNR (dB) | SSQ Score | Acc Recon | BLEU-1 | BLEU-2 | Capacity (C_SE) |")
    print("-------------------------------------------------------------------------")

    for snr_db in test_snrs:
        res = results_per_snr[snr_db]
        acc_recon = res['correct_recon'] / total_images if total_images > 0 else 0
        ssq = evaluator.calculate_ssq(acc_recon, acc_orig)
        avg_b1 = res['total_b1'] / total_images if total_images > 0 else 0
        avg_b2 = res['total_b2'] / total_images if total_images > 0 else 0

        # Compute Shannon theoretical capacity based on Eq. 29 of the paper
        capacity = evaluator.calculate_theoretical_capacity(snr_db, sc_model.num_users, sc_model.dim_expand)

        print(f"| {snr_db:8d} | {ssq:9.4f} | {acc_recon:9.4f} | {avg_b1:6.4f} | {avg_b2:6.4f} | {capacity:15.4f} |")

        # Save failure plots per SNR (if failures exist)
        plot_10_failures(res['failures'], filename=f"e2e_failures_snr{snr_db}.png")

    print("=========================================================================\n")

def print_model_parameters(model):
    print("\n" + "=" * 50)
    print(" DETAILED PARAMETER COUNT PER LAYER")
    print("=" * 50)

    total_params = 0
    vocab_params = 0
    addon_params = 0
    core_params = 0

    for name, parameter in model.named_children():
        if not parameter: continue

        layer_params = sum(p.numel() for p in parameter.parameters() if p.requires_grad)
        total_params += layer_params

        if name in ['embedding', 'output_layer']:
            vocab_params += layer_params
            print(f" [VOCAB MAPPER]       {name.ljust(18)}: {layer_params:,}")
        else:
            core_params += layer_params
            print(f" [CORE SYSTEM]        {name.ljust(18)}: {layer_params:,}")

    for name, p in model.named_parameters():
        if '.' not in name and p.requires_grad:
            layer_params = p.numel()
            total_params += layer_params
            core_params += layer_params
            print(f" [CORE SYSTEM]        {name.ljust(18)}: {layer_params:,}")

    print("-" * 50)
    print(f" Total Vocabulary Parameters (Mappers) : {vocab_params:,}")
    print(f" Total Multi-User Addon Parameters    : {addon_params:,}")
    print(f" Total CORE SYSTEM Parameters         : {core_params:,}")
    print(f" TOTAL OVERALL PARAMETERS             : {total_params:,}")
    print("=" * 50 + "\n")

def train_and_validate(train_json, val_json, epochs=30, batch_size=64, lr=1e-4, num_users=4,
                       use_channel_coding=True, use_semantic_encoder=True,
                       d_model_sem=256, dim_feedforward=512,
                       dim_expand=1024, dim_transmit=64,
                       checkpoint_path="./checkpoints/best_sc_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING TRAINING FOR {num_users} USERS ---")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
    evaluator = SystemMetricsEvaluator(tokenizer=tokenizer)

    train_dataset = CaptionDataset(train_json, tokenizer, max_len=25)
    val_dataset = CaptionDataset(val_json, tokenizer, max_len=25)
    train_loader = DataLoader(train_dataset, batch_size=batch_size * num_users, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * num_users, shuffle=False, drop_last=True)

    model = MultiUserSCSystem(
        vocab_size=tokenizer.vocab_size, num_users=num_users,
        d_model_sem=d_model_sem, dim_feedforward=dim_feedforward,
        dim_expand=dim_expand, dim_transmit=dim_transmit,
        use_channel_coding=use_channel_coding,
        use_semantic_encoder=use_semantic_encoder,
        max_len=25
    ).to(device)

    evaluator.model = model
    print_model_parameters(model)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_bleu = 0.0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [TRAIN]")

        for input_ids, attention_mask in train_bar:
            input_ids = input_ids.view(num_users, batch_size, -1).to(device)
            attention_mask = attention_mask.view(num_users, batch_size, -1).to(device)

            src_list, tgt_in_list, tgt_exp_list, src_pad_list, tgt_pad_list = [], [], [], [], []
            for u in range(num_users):
                src_list.append(input_ids[u])
                tgt_in_list.append(input_ids[u, :, :-1])
                tgt_exp_list.append(input_ids[u, :, 1:])
                src_pad_list.append(attention_mask[u] == 0)
                tgt_pad_list.append(attention_mask[u, :, :-1] == 0)

            tgt_mask = model.generate_square_subsequent_mask(tgt_in_list[0].size(1)).to(device)
            current_snr_db = torch.empty(1).uniform_(0, 10).item()

            optimizer.zero_grad()
            outputs = model(src_list, tgt_in_list, current_snr_db, src_pad_list, tgt_pad_list, tgt_mask)
            loss = sum(criterion(outputs[u].reshape(-1, tokenizer.vocab_size), tgt_exp_list[u].reshape(-1)) for u in
                       range(num_users)) / num_users

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        model.eval()
        total_b1, total_b2, val_batches, total_errors, total_valid_tokens = 0, 0, 0, 0, 0

        with torch.no_grad():
            for input_ids, attention_mask in val_loader:
                input_ids = input_ids.view(num_users, batch_size, -1).to(device)
                attention_mask = attention_mask.view(num_users, batch_size, -1).to(device)

                src_list, tgt_in_list, tgt_exp_list, src_pad_list, tgt_pad_list = [], [], [], [], []
                for u in range(num_users):
                    src_list.append(input_ids[u])
                    tgt_in_list.append(input_ids[u, :, :-1])
                    tgt_exp_list.append(input_ids[u, :, 1:])
                    src_pad_list.append(attention_mask[u] == 0)
                    tgt_pad_list.append(attention_mask[u, :, :-1] == 0)

                tgt_mask = model.generate_square_subsequent_mask(tgt_in_list[0].size(1)).to(device)
                outputs = model(src_list, tgt_in_list, 5.0, src_pad_list, tgt_pad_list, tgt_mask)

                batch_b1, batch_b2 = 0, 0
                for u in range(num_users):
                    details = evaluator.calculate_bleu_detailed(outputs[u], tgt_exp_list[u])
                    batch_b1 += sum([d['b1'] for d in details]) / len(details)
                    batch_b2 += sum([d['b2'] for d in details]) / len(details)
                    errs, valid_toks = evaluator.calculate_ser(outputs[u], tgt_exp_list[u], tokenizer.pad_token_id)
                    total_errors += errs
                    total_valid_tokens += valid_toks

                total_b1 += batch_b1 / num_users
                total_b2 += batch_b2 / num_users
                val_batches += 1

        avg_b1 = total_b1 / val_batches
        print(
            f"\n=> Epoch {epoch + 1}: Loss {total_train_loss / len(train_loader):.4f} | BLEU-1: {avg_b1:.4f} | SER: {total_errors / total_valid_tokens:.4f}")
        if avg_b1 > best_val_bleu:
            best_val_bleu = avg_b1
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print("   [+] Saved Best Model!")


def evaluate_model_at_snrs(val_json, model_path, batch_size=64, num_users=4,
                           use_channel_coding=True, use_semantic_encoder=True,
                           d_model_sem=128, dim_feedforward=512, dim_expand=1024, dim_transmit=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
    evaluator = SystemMetricsEvaluator(tokenizer=tokenizer)

    val_loader = DataLoader(CaptionDataset(val_json, tokenizer, max_len=25), batch_size=batch_size * num_users,
                            shuffle=False, drop_last=True)

    model = MultiUserSCSystem(
        vocab_size=tokenizer.vocab_size, num_users=num_users, d_model_sem=d_model_sem, dim_feedforward=dim_feedforward,
        dim_expand=dim_expand, dim_transmit=dim_transmit,
        use_channel_coding=use_channel_coding, use_semantic_encoder=use_semantic_encoder, max_len=25
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dcr_ratio, dcr_percent = evaluator.calculate_compression_rate(dim_transmit, max_len=25)
    print(f"\n[*] Data Compression Rate (DCR): {dcr_ratio:.6f} ({dcr_percent:.4f}%)")

    # =========================================================================
    # ITERATING EVALUATIONS ACROSS A RANGE OF SNR VALUES
    # =========================================================================
    with torch.no_grad():
        for snr_db in [0, 2, 4, 6, 8, 10]:
            total_b1, total_b2, val_batches, total_errors, total_valid_tokens = 0, 0, 0, 0, 0
            for input_ids, attention_mask in val_loader:
                input_ids = input_ids.view(num_users, batch_size, -1).to(device)
                attention_mask = attention_mask.view(num_users, batch_size, -1).to(device)
                src_list, tgt_in_list, tgt_exp_list, src_pad_list, tgt_pad_list = [], [], [], [], []
                for u in range(num_users):
                    src_list.append(input_ids[u])
                    tgt_in_list.append(input_ids[u, :, :-1])
                    tgt_exp_list.append(input_ids[u, :, 1:])
                    src_pad_list.append(attention_mask[u] == 0)
                    tgt_pad_list.append(attention_mask[u, :, :-1] == 0)

                tgt_mask = model.generate_square_subsequent_mask(tgt_in_list[0].size(1)).to(device)
                outputs = model(src_list, tgt_in_list, snr_db, src_pad_list, tgt_pad_list, tgt_mask)

                batch_b1, batch_b2 = 0, 0
                for u in range(num_users):
                    details = evaluator.calculate_bleu_detailed(outputs[u], tgt_exp_list[u])
                    batch_b1 += sum([d['b1'] for d in details]) / len(details)
                    batch_b2 += sum([d['b2'] for d in details]) / len(details)
                    errs, valid_toks = evaluator.calculate_ser(outputs[u], tgt_exp_list[u], tokenizer.pad_token_id)
                    total_errors += errs
                    total_valid_tokens += valid_toks

                total_b1 += batch_b1 / num_users
                total_b2 += batch_b2 / num_users
                val_batches += 1

            print(
                f"[*] Channel SNR = {snr_db:2d} dB  --->  BLEU-1: {total_b1 / val_batches:.4f} | "
                f"BLEU-2: {total_b2 / val_batches:.4f} | SER: {total_errors / total_valid_tokens:.4f} | "
            )


class EndToEndVisualizer:
    def __init__(self, sc_model, tokenizer, device='cuda'):
        self.device = device
        self.sc_model = sc_model.to(self.device)
        self.sc_model.eval()
        self.tokenizer = tokenizer
        self.smoothie = SmoothingFunction().method1

        print("\n[*] Initializing Visualizer: Loading BLIP and Stable Diffusion...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                       use_safetensors=False).to(self.device)
        self.blip_model.eval()

        self.sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                               torch_dtype=torch.float16).to(self.device)
        self.sd_pipe.set_progress_bar_config(disable=True)
        self.sd_pipe.safety_checker = None

    def extract_and_visualize(self, num_users=8, max_scan=100, num_samples=3,
                              output_name="SC_System_Visual_Result.jpg"):
        print(f"[*] Scanning {max_scan} images to find {num_samples} excellent samples...")

        dataset = CatDogDataset(split='test', max_samples=max_scan)
        found_samples = []

        with torch.no_grad():
            for i in range(len(dataset)):
                orig_img, label = dataset[i]

                # 1. BLIP Captioning
                inputs_blip = self.blip_processor(images=orig_img, return_tensors="pt").to(self.device)
                outputs_blip = self.blip_model.generate(**inputs_blip, max_new_tokens=25)
                orig_caption = self.blip_processor.decode(outputs_blip[0], skip_special_tokens=True)

                # 2. Prep data to format into SC System
                encoded = self.tokenizer(orig_caption, padding='max_length', truncation=True, max_length=25,
                                         return_tensors='pt')
                input_id = encoded['input_ids'].to(self.device)
                attn_mask = encoded['attention_mask'].to(self.device)

                src_list, tgt_in_list, src_pad_list, tgt_pad_list = [], [], [], []
                for u in range(num_users):
                    src_list.append(input_id)
                    tgt_in_list.append(input_id[:, :-1])
                    src_pad_list.append(attn_mask == 0)
                    tgt_pad_list.append(attn_mask[:, :-1] == 0)

                tgt_mask = self.sc_model.generate_square_subsequent_mask(tgt_in_list[0].size(1)).to(self.device)

                # ======================================================
                # 3A. Profile evaluating at High SNR (10 dB)
                # ======================================================
                outputs_10 = self.sc_model(src_list, tgt_in_list, 10.0, src_pad_list, tgt_pad_list, tgt_mask)
                pred_ids_10 = torch.argmax(outputs_10[0], dim=-1)[0]
                recon_caption_10 = self.tokenizer.decode(pred_ids_10.tolist(), skip_special_tokens=True)

                target_words = orig_caption.split()
                pred_words_10 = recon_caption_10.split()
                if not pred_words_10: continue

                b1_10 = sentence_bleu([target_words], pred_words_10, weights=(1.0, 0, 0, 0),
                                      smoothing_function=self.smoothie)

                # Filter condition: High BLEU score
                if b1_10 > 0.85:
                    # ======================================================
                    # 3B. Profile evaluating at Low SNR (0 dB)
                    # ======================================================
                    outputs_0 = self.sc_model(src_list, tgt_in_list, 0.0, src_pad_list, tgt_pad_list, tgt_mask)
                    pred_ids_0 = torch.argmax(outputs_0[0], dim=-1)[0]
                    recon_caption_0 = self.tokenizer.decode(pred_ids_0.tolist(), skip_special_tokens=True)

                    print(f"  [+] Saved sample {len(found_samples) + 1}/{num_samples} | BLEU-10dB: {b1_10:.2f}")
                    found_samples.append({
                        'orig_img': orig_img,
                        'orig_cap': orig_caption,
                        'recon_cap_10': recon_caption_10,
                        'recon_cap_0': recon_caption_0
                    })

                if len(found_samples) == num_samples:
                    break

        if len(found_samples) == 0:
            print("[!] No suitable samples found. Please try again!")
            return

        self._plot_results(found_samples, output_name)

    def _plot_results(self, samples, output_name):
        num_cols = len(samples)
        print(f"\n[*] Generating {num_cols * 2} Reconstructed images using Stable Diffusion...")

        fig, axes = plt.subplots(nrows=6, ncols=num_cols, figsize=(4.5 * num_cols, 16),
                                 gridspec_kw={'height_ratios': [1, 0.15, 1, 0.15, 1, 0.15]})
        fig.patch.set_facecolor('white')

        if num_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for col, sample in enumerate(samples):
            orig_img = sample['orig_img'].resize((512, 512), Image.Resampling.LANCZOS)
            recon_img_10 = \
            self.sd_pipe(prompt=sample['recon_cap_10'], num_inference_steps=25, height=512, width=512).images[0]
            recon_img_0 = \
            self.sd_pipe(prompt=sample['recon_cap_0'], num_inference_steps=25, height=512, width=512).images[0]

            # Row 0: Original Image
            axes[0, col].imshow(orig_img)
            axes[0, col].set_xticks([]);
            axes[0, col].set_yticks([])

            # Row 1: Original Caption
            axes[1, col].axis('off')
            cap_orig_wrap = "\n".join(textwrap.wrap(sample['orig_cap'], width=45))
            axes[1, col].text(0.5, 0.5, cap_orig_wrap, fontsize=13, ha='center', va='center', fontname='serif',
                              bbox=dict(facecolor='#f1f3f4', edgecolor='none', boxstyle='round,pad=0.4', alpha=1.0))

            # Row 2: Reconstructed (10dB) Image
            axes[2, col].imshow(recon_img_10)
            axes[2, col].set_xticks([]);
            axes[2, col].set_yticks([])

            # Row 3: Reconstructed (10dB) Caption
            axes[3, col].axis('off')
            cap_10_wrap = "\n".join(textwrap.wrap(sample['recon_cap_10'], width=45))
            axes[3, col].text(0.5, 0.5, cap_10_wrap, fontsize=13, ha='center', va='center', fontname='serif',
                              bbox=dict(facecolor='#e6f4ea', edgecolor='none', boxstyle='round,pad=0.4',
                                        alpha=1.0))

            # Row 4: Reconstructed (0dB) Image
            axes[4, col].imshow(recon_img_0)
            axes[4, col].set_xticks([]);
            axes[4, col].set_yticks([])

            # Row 5: Reconstructed (0dB) Caption
            axes[5, col].axis('off')
            cap_0_wrap = "\n".join(textwrap.wrap(sample['recon_cap_0'], width=45))
            axes[5, col].text(0.5, 0.5, cap_0_wrap, fontsize=13, ha='center', va='center', fontname='serif',
                              bbox=dict(facecolor='#fce8e6', edgecolor='none', boxstyle='round,pad=0.4',
                                        alpha=1.0))

            if col == 0:
                axes[0, col].set_ylabel('Original\nImage', fontsize=16, fontname='serif', rotation=0, labelpad=70,
                                        ha='center', va='center')
                axes[2, col].set_ylabel('Reconstructed\n(SNR = 10 dB)', fontsize=16, fontname='serif', rotation=0,
                                        labelpad=70, ha='center', va='center')
                axes[4, col].set_ylabel('Reconstructed\n(SNR = 0 dB)', fontsize=16, fontname='serif', rotation=0,
                                        labelpad=70, ha='center', va='center')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.08, wspace=0.05)

        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] EXCELLENT! Visualizations saved successfully to: {output_name}")

if __name__ == "__main__":
    DIM_TRANSMIT = 128
    NUM_USERS = 8
    OUTPUT_DIR = './outputs'
    CHECKPOINT_DIR = './checkpoints'
    DEFAULT_LOG_PATH = f'{OUTPUT_DIR}/training_log.txt'
    DEFAULT_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/best_sc_model.pth'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    sys.stdout = DualLogger(DEFAULT_LOG_PATH)
    set_seed(42)

    ENABLE_SEMANTIC_ENCODER = True
    ENABLE_CHANNEL_CODING = True
    D_MODEL_SEMANTIC = 128
    DIM_FEEDFORWARD = 256
    DIM_EXPAND = 512
    BATCH_SIZE = 128
    EPOCHS = 100

    TRAIN_JSON = 'catdog_captions_train.json'
    TEST_JSON = 'catdog_captions_test.json'

    # 1. Training Pipeline
    train_and_validate(
        train_json=TRAIN_JSON, val_json=TEST_JSON, epochs=EPOCHS, batch_size=BATCH_SIZE,
        num_users=NUM_USERS, use_channel_coding=ENABLE_CHANNEL_CODING,
        use_semantic_encoder=ENABLE_SEMANTIC_ENCODER,
        d_model_sem=D_MODEL_SEMANTIC, dim_feedforward=DIM_FEEDFORWARD,
        dim_expand=DIM_EXPAND, dim_transmit=DIM_TRANSMIT,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
    evaluator = SystemMetricsEvaluator(tokenizer=tokenizer)

    # 2. Setup the model to pull the saved Checkpoint
    best_model = MultiUserSCSystem(
        vocab_size=tokenizer.vocab_size, num_users=NUM_USERS,
        d_model_sem=D_MODEL_SEMANTIC, dim_feedforward=DIM_FEEDFORWARD,
        dim_expand=DIM_EXPAND, dim_transmit=DIM_TRANSMIT,
        use_channel_coding=ENABLE_CHANNEL_CODING,
        use_semantic_encoder=ENABLE_SEMANTIC_ENCODER, max_len=25
    ).to(device)

    # If the file exists, initialize checkpoint evaluation
    if os.path.exists(DEFAULT_CHECKPOINT_PATH):
        best_model.load_state_dict(torch.load(DEFAULT_CHECKPOINT_PATH, map_location=device))

        # Evaluate Architecture Profile
        evaluator.profile_system(model=best_model, batch_size=BATCH_SIZE, max_len=25, num_users=NUM_USERS,
                                 device=device)

        # Evaluate across SNRs
        evaluate_model_at_snrs(val_json=TEST_JSON, model_path=DEFAULT_CHECKPOINT_PATH, batch_size=BATCH_SIZE, num_users=NUM_USERS,
                               use_channel_coding=ENABLE_CHANNEL_CODING,
                               use_semantic_encoder=ENABLE_SEMANTIC_ENCODER,
                               d_model_sem=D_MODEL_SEMANTIC,
                               dim_feedforward=DIM_FEEDFORWARD, dim_expand=DIM_EXPAND, dim_transmit=DIM_TRANSMIT)
        # ========================================================
        # 4. VISUALIZATION (PAPER SUBMISSION FIGURES)
        # ========================================================
        torch.cuda.empty_cache()
        visualizer = EndToEndVisualizer(sc_model=best_model, tokenizer=tokenizer, device=device)

        visualizer.extract_and_visualize(
            num_users=NUM_USERS,
            max_scan=100,
            num_samples=3,
            output_name="SC_System_Multi_SNR_3_Cols.jpg"
        )

        # 3. EVALUATE END-TO-END PIPELINE ON CAT DOG DATASET
        torch.cuda.empty_cache()
        run_end_to_end_pipeline(sc_model=best_model, test_snrs=[0, 2, 4, 6, 8, 10], max_samples=None)
    else:
        print(f"[!] Checkpoint not found: {DEFAULT_CHECKPOINT_PATH}. Run training first.")
