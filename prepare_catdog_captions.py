"""Cat vs Dog caption-generation and image-reconstruction pipeline.

This script prepares the data used by the semantic communication model:
1. Download Cats vs Dogs from Hugging Face.
2. Generate BLIP captions and save them as JSON files.
3. Optionally reconstruct images from captions with Stable Diffusion.
4. Optionally evaluate semantic similarity with CLIP.
"""
import random
import sys
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import textwrap

# Required for loading the Cats vs Dogs dataset from Hugging Face.
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install the datasets library using: pip install datasets")

# Download the NLTK tokenizer resource if it is not already available.
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import nltk
    nltk.download('punkt')


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
# 0. DUAL LOGGER FOR SAVING TO TXT
# ==========================================
class DualLogger(object):
    """
    Writes to sys.stdout to print to the screen and saves to a txt file simultaneously.
    Filters out the '\r' character from tqdm progress bars to keep the txt file clean.
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
        """HuggingFace needs this function to check if it should print colors to the console."""
        return False


# ==========================================
# 1. DATASET WRAPPER (CATS VS DOGS)
# ==========================================
class CatDogDataset(Dataset):
    """Wrapper to make the HuggingFace dataset compatible with PyTorch DataLoader."""
    def __init__(self, split='test', max_samples=None):
        # Load dataset microsoft/cats_vs_dogs
        print(f"[*] Loading dataset microsoft/cats_vs_dogs...")
        ds = load_dataset("microsoft/cats_vs_dogs", split="train", trust_remote_code=True)

        # Split 80% train, 20% test (set fixed seed for consistency)
        ds = ds.train_test_split(test_size=0.2, seed=42)
        self.data = ds[split]

        # LIMIT NUMBER OF SAMPLES (IF CONFIGURED)
        if max_samples is not None:
            actual_samples = min(max_samples, len(self.data))
            self.data = self.data.select(range(actual_samples))
            print(f"[*] Reduced the {split.upper()} dataset to {actual_samples} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image'].convert('RGB')
        label = item['labels']  # 0 for cat, 1 for dog
        return img, label


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels


def save_category_plot(samples, category_name, filename):
    """Draw and save visual analysis plots comparing Original vs. Reconstructed images."""
    if not samples:
        print(f"No samples found for category {category_name}")
        return

    classes = ['Cat', 'Dog']
    fig, axes = plt.subplots(nrows=len(samples), ncols=3, figsize=(18, 4.5 * len(samples)))
    fig.suptitle(f"Category: {category_name}", fontsize=20, fontweight='bold')

    for i, sample in enumerate(samples):
        ax_orig = axes[i, 0] if len(samples) > 1 else axes[0]
        ax_recon = axes[i, 1] if len(samples) > 1 else axes[1]
        ax_text = axes[i, 2] if len(samples) > 1 else axes[2]

        orig_img_256 = sample['orig_img'].resize((256, 256), Image.NEAREST)
        recon_img_256 = sample['recon_img'].resize((256, 256), Image.NEAREST)

        ax_orig.imshow(orig_img_256)
        true_cls = classes[sample['true_label']]
        pred_orig_cls = classes[sample['pred_orig']]
        ax_orig.set_title(f"Original\nTrue Label: {true_cls}\nPred: {pred_orig_cls}", fontsize=14)
        ax_orig.axis('off')

        ax_recon.imshow(recon_img_256)
        pred_recon_cls = classes[sample['pred_recon']]
        ax_recon.set_title(f"Reconstructed\nPred: {pred_recon_cls}", fontsize=14)
        ax_recon.axis('off')

        ax_text.axis('off')
        wrapped_text = textwrap.fill(sample['caption'], width=45)
        ax_text.text(0.05, 0.5, f"Caption:\n\n{wrapped_text}", fontsize=14, verticalalignment='center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92 if len(samples) > 1 else 0.85)
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved analysis plot to: {filename}")


# ==========================================
# 2. DATA PIPELINE STEP 1: CAPTION EXTRACTION (BLIP)
# ==========================================
def run_captioning(split='test', batch_size=64, max_samples=None):
    """
    Extracts semantic textual features (captions) from original images using BLIP.
    The resulting JSON acts as the ground-truth semantic payload for the system.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    suffix = f"_{max_samples}" if max_samples is not None else ""
    output_json = f'catdog_captions_{split}{suffix}.json'

    if os.path.exists(output_json):
        print(f"[*] Found {output_json}. Skipping Captioning step for the {split.upper()} split.")
        return output_json

    print(f"\n--- STARTING CAPTION EXTRACTION FOR {split.upper()} SPLIT ---")
    dataset = CatDogDataset(split=split, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True
    ).to(device)
    model.eval()

    captions_data = []
    global_idx = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Captioning CatDog {split}"):
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=30)
            captions = processor.batch_decode(outputs, skip_special_tokens=True)

            for label, caption in zip(labels, captions):
                captions_data.append({"index": global_idx, "label": label, "caption": caption})
                global_idx += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, ensure_ascii=False, indent=4)
    print(f"[+] Finished generating {output_json}\n")

    del model
    torch.cuda.empty_cache()
    return output_json


# ==========================================
# 3. DATA PIPELINE STEP 2: IMAGE RECONSTRUCTION (STABLE DIFFUSION)
# ==========================================
def run_reconstruction(split='test', batch_size=16, max_samples=None):
    """
    Simulates the receiver end-task. Generates images back from the text payload 
    using Stable Diffusion to test visually if semantic meaning was preserved.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    suffix = f"_{max_samples}" if max_samples is not None else ""
    input_json = f'catdog_captions_{split}{suffix}.json'
    output_dir = f'./reconstructed_images_catdog_{split}{suffix}'
    metadata_file = f'reconstructed_metadata_catdog_{split}{suffix}.json'

    if os.path.exists(metadata_file):
        print(f"[*] Found {metadata_file}. Skipping Reconstruction step for the {split.upper()} split.")
        return metadata_file

    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Error: {input_json} not found. Run the Captioning step first!")

    print(f"\n--- STARTING IMAGE RECONSTRUCTION FOR {split.upper()} SPLIT ---")
    model_id = "sd-legacy/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    with open(input_json, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    reconstructed_data = []
    total_images = len(captions_data)

    for i in tqdm(range(0, total_images, batch_size), desc=f"Reconstructing {split}"):
        batch_items = captions_data[i: i + batch_size]
        captions = [item['caption'] for item in batch_items]

        with torch.no_grad():
            images = pipe(prompt=captions, num_inference_steps=20, height=512, width=512).images

        for j, item in enumerate(batch_items):
            img_name = f"recon_catdog_{item['index']}_label_{item['label']}.png"
            img_path = os.path.join(output_dir, img_name)

            images[j].resize((256, 256), Image.BICUBIC).save(img_path)

            reconstructed_data.append({
                "index": item['index'],
                "label": item['label'],
                "caption": captions[j],
                "recon_img_path": img_path
            })

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(reconstructed_data, f, indent=4)
    print(f"[+] Finished generating {metadata_file}\n")

    del pipe
    torch.cuda.empty_cache()
    return metadata_file


# ==========================================
# 4. METRIC COMPUTATION: SEMANTIC SIMILARITY EVALUATION (CLIP ZERO-SHOT)
# ==========================================
def run_evaluation(split='test', max_samples=None):
    """
    Evaluates Semantic Similarity Quality (SSQ). 
    Uses CLIP to determine if the reconstructed image retains the semantic class 
    (Cat/Dog) from the original image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    suffix = f"_{max_samples}" if max_samples is not None else ""
    recon_metadata = f'reconstructed_metadata_catdog_{split}{suffix}.json'
    eval_dir = f'evaluation_samples_catdog{suffix}'

    if not os.path.exists(recon_metadata):
        print(f"Error: Missing file {recon_metadata} for evaluation.")
        return

    print(f"\n--- STARTING SSQ EVALUATION FOR {split.upper()} SPLIT (CLIP) ---")

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    classes_catdog = ['cat', 'dog']
    text_prompts = [f"a photo of a {c}" for c in classes_catdog]

    original_dataset = CatDogDataset(split=split, max_samples=max_samples)

    with open(recon_metadata, 'r') as f:
        recon_data = json.load(f)

    samples_cc, samples_cw, samples_wc, samples_ww = [], [], [], []
    correct_original, correct_recon = 0, 0

    with torch.no_grad():
        for item in tqdm(recon_data, desc=f"Evaluating {split}"):
            idx = item['index']
            true_label = item['label']

            orig_img_pil, _ = original_dataset[idx]
            recon_img_pil = Image.open(item['recon_img_path']).convert('RGB')

            inputs_orig = clip_processor(text=text_prompts, images=orig_img_pil, return_tensors="pt", padding=True).to(device)
            pred_orig = clip_model(**inputs_orig).logits_per_image.argmax(dim=1).item()

            inputs_recon = clip_processor(text=text_prompts, images=recon_img_pil, return_tensors="pt", padding=True).to(device)
            pred_recon = clip_model(**inputs_recon).logits_per_image.argmax(dim=1).item()

            is_orig_correct = (pred_orig == true_label)
            is_recon_correct = (pred_recon == true_label)

            if is_orig_correct: correct_original += 1
            if is_recon_correct: correct_recon += 1

            sample_data = {
                'orig_img': orig_img_pil,
                'recon_img': recon_img_pil,
                'caption': item['caption'],
                'true_label': true_label,
                'pred_orig': pred_orig,
                'pred_recon': pred_recon
            }
            if is_orig_correct and is_recon_correct and len(samples_cc) < 5:
                samples_cc.append(sample_data)
            elif is_orig_correct and not is_recon_correct and len(samples_cw) < 5:
                samples_cw.append(sample_data)
            elif not is_orig_correct and is_recon_correct and len(samples_wc) < 5:
                samples_wc.append(sample_data)
            elif not is_orig_correct and not is_recon_correct and len(samples_ww) < 5:
                samples_ww.append(sample_data)

    total = len(recon_data)
    acc_orig = correct_original / total if total > 0 else 0
    acc_recon = correct_recon / total if total > 0 else 0
    ssq = acc_recon / acc_orig if acc_orig > 0 else 0

    print(f"\n[RESULTS FOR CAT-DOG {split.upper()}]")
    print(f"Total samples: {total}")
    print(f"Original Image Accuracy (CLIP): {acc_orig:.4f} | Reconstructed Image Accuracy (CLIP): {acc_recon:.4f}")
    print(f"SSQ: {ssq:.4f}\n")

    os.makedirs(eval_dir, exist_ok=True)
    save_category_plot(samples_cc, f"{split.upper()} - Correct Orig & Correct Recon",
                       f"{eval_dir}/sample_{split}_CC.png")
    save_category_plot(samples_cw, f"{split.upper()} - Correct Orig & Wrong Recon",
                       f"{eval_dir}/sample_{split}_CW.png")
    save_category_plot(samples_wc, f"{split.upper()} - Wrong Orig & Correct Recon",
                       f"{eval_dir}/sample_{split}_WC.png")
    save_category_plot(samples_ww, f"{split.upper()} - Wrong Orig & Wrong Recon",
                       f"{eval_dir}/sample_{split}_WW.png")

    del clip_model
    torch.cuda.empty_cache()


# ==========================================
# 5. MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    sys.stdout = DualLogger("catdog_pipeline_log.txt")

    # ==============================================================
    # RUN CONFIGURATION:
    # If MAX_TEST_SAMPLES = None -> Run FULL TRAIN and TEST dataset.
    # If MAX_TEST_SAMPLES = (number) -> Only run TEST split with that number of samples.
    # ==============================================================
    MAX_TEST_SAMPLES = 10

    if MAX_TEST_SAMPLES is None:
        splits_to_run = ['train', 'test']
        print(f"[*] FULL DATASET MODE: Configured to run entire TRAIN and TEST splits.")
    else:
        splits_to_run = ['test']
        print(f"[*] QUICK TEST MODE: Running TEST split only, limited to {MAX_TEST_SAMPLES} samples.")

    for current_split in splits_to_run:
        # If running train split, run full dataset. If test split, apply the sample limit (if any).
        current_max_samples = MAX_TEST_SAMPLES if current_split == 'test' else None

        print("\n" + "=" * 60)
        print(f" STARTING END-TO-END PIPELINE FOR CATS VS DOGS: {current_split.upper()}")
        print("=" * 60)

        # Step 1: Generate BLIP captions and save catdog_captions_<split>.json.
        run_captioning(split=current_split, batch_size=64, max_samples=current_max_samples)

        # Step 2: Reconstruct images from generated captions with Stable Diffusion.
        run_reconstruction(split=current_split, batch_size=16, max_samples=current_max_samples)

        # Step 3: Evaluate original/reconstructed images with CLIP zero-shot classification.
        run_evaluation(split=current_split, max_samples=current_max_samples)
