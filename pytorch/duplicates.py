import os
import torch
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil

# --- CONFIGURATION ---
DATASET_PATH = "./nodup_data/"
# The new quarantine folder name as requested
QUARANTINE_BASE_PATH = "./reviewdup/"
MODEL_NAME = "mobilenetv3_small_100.lamb_in1k"
SIMILARITY_THRESHOLD = 0.995
BATCH_SIZE = 32


def get_model_and_transforms():
    """Loads the pre-trained model and its corresponding transforms just once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    return model, transform, device


def get_image_paths(folder_path):
    """Finds all valid image paths within a specific folder."""
    paths = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_exts):
            paths.append(os.path.join(folder_path, filename))
    return paths


@torch.no_grad()
def get_embeddings(model, transform, device, image_paths):
    """Generates feature vectors (embeddings) for a list of images."""
    if not image_paths:
        return np.array([])

    embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="  Generating Embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(transform(img))
            except (IOError, OSError) as e:
                print(f"\n  ⚠️ Skipping corrupted image: {path} ({e})")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        batch_embeddings = model(batch_tensor)
        embeddings.append(batch_embeddings.cpu().numpy())

    if not embeddings:
        return np.array([])

    return np.vstack(embeddings)


def process_single_folder(model, transform, device, class_path, quarantine_path):
    """Finds and moves near-duplicates within a single class folder."""

    # 1. Get image paths from the current class folder
    image_paths = sorted(get_image_paths(class_path))

    if len(image_paths) < 2:
        print("  Not enough images to compare.")
        return

    # 2. Generate Embeddings for all images in the folder
    embeddings = get_embeddings(model, transform, device, image_paths)
    if embeddings.shape[0] < 2:
        print("  Not enough valid embeddings to compare.")
        return

    # 3. Calculate Cosine Similarity
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)

    # 4. Identify files to move
    files_to_move = set()
    for i in range(similarity_matrix.shape[0]):
        duplicate_indices = np.where(similarity_matrix[i] > SIMILARITY_THRESHOLD)[0]
        for idx in duplicate_indices:
            if i < idx:
                files_to_move.add(image_paths[idx])

    if not files_to_move:
        print("  ✅ No near-duplicates found in this class.")
        return

    # 5. Move the identified duplicates
    os.makedirs(quarantine_path, exist_ok=True)
    print(f"  Found {len(files_to_move)} similar files. Moving them to {quarantine_path}")

    for file_path in files_to_move:
        try:
            # Move file to the corresponding class subfolder in quarantine
            shutil.move(file_path, quarantine_path)
        except Exception as e:
            print(f"  ❌ Failed to move {file_path}: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Per-Class Near-Duplicate Image Cleaner ---")

    # Load the model once to be reused for all folders
    model, transform, device = get_model_and_transforms()

    # Find all the class subfolders in the main dataset directory
    try:
        class_dirs = [d for d in os.scandir(DATASET_PATH) if d.is_dir()]
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset path not found at '{DATASET_PATH}'. Please check the path.")
        exit()

    # Process each class folder individually
    for class_dir in class_dirs:
        print(f"\nProcessing class: {class_dir.name}")
        class_quarantine_path = os.path.join(QUARANTINE_BASE_PATH, class_dir.name)
        process_single_folder(model, transform, device, class_dir.path, class_quarantine_path)

    print("\n--- All classes processed. ---")
