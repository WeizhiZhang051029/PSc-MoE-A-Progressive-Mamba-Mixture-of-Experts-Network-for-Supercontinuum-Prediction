"""Data loading utilities.

Responsibilities:
  - Read individual .mat files with load_single_mat.
  - Preload datasets with multiple threads via SCDataset.
  - Build DataLoader instances with create_loader.
  - Set random seeds with set_seed.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

import config


MAT_KEY = "new_lIW"
DEFAULT_FIBER_STEPS = 200


def infer_data_shape(data_folder: str, mat_key: str = MAT_KEY) -> tuple[int, int]:
    """Read the first valid .mat file in a data directory and return its 2D spectrum shape."""
    for file_path in collect_mat_files(data_folder):
        try:
            mat_data = sio.loadmat(file_path)
            if mat_key not in mat_data:
                continue
            data = np.asarray(mat_data[mat_key])
            if data.ndim == 2:
                return int(data.shape[0]), int(data.shape[1])
            print(f"[Warning] {file_path} is not a 2D spectrogram; shape = {data.shape}, skipping")
        except Exception as e:
            print(f"[Warning] Failed to infer shape from {file_path}: {e}")
    raise RuntimeError(f"No valid 2D spectrogram found in the data directory: {data_folder}")


def sync_data_shape_from_folder(data_folder: str) -> tuple[int, int]:
    """Synchronize propagation steps and spectrum length from the actual dataset shape."""
    fiber_steps, spectrum_size = infer_data_shape(data_folder)
    old_fiber_steps = getattr(config, "FIBER_STEPS", DEFAULT_FIBER_STEPS)
    old_output_size = getattr(config, "OUTPUT_SIZE", spectrum_size)
    config.FIBER_STEPS = fiber_steps
    config.OUTPUT_SIZE = spectrum_size
    if config.USE_FIRST_STEP_INPUT:
        config.INPUT_SIZE = spectrum_size
    if old_fiber_steps != fiber_steps or old_output_size != spectrum_size:
        print(
            f"[Data Shape] Automatically configured from dataset: propagation_steps={fiber_steps}, spectrum_size={spectrum_size}"
        )
    return fiber_steps, spectrum_size


# ============================================================
# Random seed
# ============================================================
def set_seed(seed: int) -> None:
    """Fix all random sources for reproducible experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ============================================================
# Filename parameter parsing
# ============================================================
def parse_params_from_filename(file_path: str):
    """Parse three physical parameters from the filename and return a FloatTensor with shape (3,).
    
    Filename format: {pump_power}_{fiber_length}_{gamma}.mat
    Example: 3200_1000_1.000e-13.mat -> [3200.0, 1000.0, 1.000]
    
    For gamma, only the mantissa is used and the scientific-notation exponent is discarded.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]  # Remove .mat suffix
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    try:
        p1 = float(parts[0])   # pump_power, for example 3200
        p2 = float(parts[1])   # fiber_length, for example 1000
        # gamma looks like 1.000e-13; keep only the mantissa
        gamma_str = parts[2]   # Example: 1.000e-13
        mantissa = float(gamma_str.split('e')[0])   # 1.000
        return torch.tensor([p1, p2, mantissa], dtype=torch.float32)
    except Exception as e:
        print(f"[Warning] Failed to parse filename {file_path}: {e}")
        return None


# ============================================================
# .mat file path collection
# ============================================================
def collect_mat_files(data_folder: str):
    """Collect .mat file paths from a data directory and return a sorted path list.
    
    Two dataset layouts are supported:
      1. .mat files are stored directly under train/val.
      2. train/val contains lambda-prefixed subfolders with .mat files.
    
    All paths are sorted to keep experiments reproducible.
    """
    if not os.path.isdir(data_folder):
        print(f"[Warning] Data directory does not exist: {data_folder}")
        return []

    # First priority: find .mat files directly under the train/val root directory
    root_mat_files = sorted(
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.lower().endswith('.mat') and os.path.isfile(os.path.join(data_folder, f))
    )
    if root_mat_files:
        print(f"[Data Loading] Found {len(root_mat_files)} .mat file(s) directly in the root directory")
        return root_mat_files

    # Second priority: if no root-level .mat files exist, search lambda-prefixed subfolders
    lambda_dirs = sorted(
        os.path.join(data_folder, d)
        for d in os.listdir(data_folder)
        if d.startswith('λ') and os.path.isdir(os.path.join(data_folder, d))
    )
    if not lambda_dirs:
        print(f"[Warning] No .mat files found in {data_folder} root, and no λ-prefixed subdirectories detected")
        return []

    all_files = []
    for lambda_dir in lambda_dirs:
        mat_files = sorted(
            os.path.join(lambda_dir, f)
            for f in os.listdir(lambda_dir)
            if f.lower().endswith('.mat') and os.path.isfile(os.path.join(lambda_dir, f))
        )
        print(f"[Data Loading] Entering subdirectory {lambda_dir}, found {len(mat_files)} .mat file(s)")
        all_files.extend(mat_files)

    all_files = sorted(all_files)
    if not all_files:
        print(f"[Warning] λ-prefixed subdirectories exist, but no .mat files found in {data_folder}")
    return all_files


# ============================================================
# Single-file reading
# ============================================================
def load_single_mat(file_path: str):
    """Read one .mat file.
    
    When config.USE_FIRST_STEP_INPUT is True, return only target [200, 251].
    Otherwise return (params [3], target [200, 251]) with params parsed from the filename.
    Return None if reading fails, the shape is invalid, or filename parsing fails.
    """
    try:
        mat_data = sio.loadmat(file_path)
        if MAT_KEY not in mat_data:
            print(f"[Warning] Field '{MAT_KEY}' not found in {file_path}")
            return None
        lIW = mat_data[MAT_KEY]
        expected_shape = (
            getattr(config, "FIBER_STEPS", DEFAULT_FIBER_STEPS),
            config.OUTPUT_SIZE,
        )
        if lIW.shape != expected_shape:
            print(f"[Warning] Shape mismatch for {file_path}: got {lIW.shape}, expected {expected_shape}, skipping")
            return None
        target = torch.from_numpy(np.ascontiguousarray(lIW)).float()
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        return None

    if config.USE_FIRST_STEP_INPUT:
        # First-step spectrum mode: return the full spectrum map; inputs are taken from target[:, 0, :] during training
        return target
    else:
        # Physical-parameter mode: parse three parameters from the filename
        params = parse_params_from_filename(file_path)
        if params is None:
            return None
        return params, target


# ============================================================
# Dataset
# ============================================================
class SCDataset(Dataset):
    """Supercontinuum dataset.
    
    All .mat files are preloaded into memory with multiple threads at startup, so training reads samples directly from memory and avoids disk I/O bottlenecks.
    """

    def __init__(self, data_folder: str, num_threads: int = 8):
        self.data_folder = data_folder
        self.num_threads = max(1, num_threads)
        sync_data_shape_from_folder(data_folder)

        # Collect and sort all .mat file paths to keep ordering reproducible
        # Prefer .mat files directly under train/val; if none exist at the root,
        # continue reading .mat files from lambda-prefixed subfolders.
        all_files = collect_mat_files(data_folder)

        print(f"[Data Loading] {data_folder}  {len(all_files)} file(s), reading with {self.num_threads} thread(s)...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(load_single_mat, all_files))
        self.samples = [x for x in results if x is not None]
        print(f"[Data Loading] Completed: {len(self.samples)} valid sample(s) loaded in {time.time() - t0:.2f}s")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Return (params [3], target [200, 251])
        return self.samples[idx]


# ============================================================
# DataLoader factory
# ============================================================
def create_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a DataLoader and configure multiprocessing prefetch options when needed.
    On Windows, num_workers is forced to 0 to avoid multiprocessing issues.
    """
    kwargs = dict(
        dataset    = dataset,
        batch_size = batch_size,
        shuffle    = shuffle,
        pin_memory = torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs.update(
            num_workers      = num_workers,
            persistent_workers = True,
            prefetch_factor  = config.PREFETCH_FACTOR,
        )
    return DataLoader(**kwargs)


# ============================================================
# Public entry point: build train and validation loaders
# ============================================================
def build_loaders():
    """Build training and validation DataLoaders."""
    train_dataset = SCDataset(
        config.TRAIN_DATA_FOLDER,
        num_threads=config.PRELOAD_NUM_THREADS,
    )
    val_dataset = SCDataset(
        config.VAL_DATA_FOLDER,
        num_threads=config.PRELOAD_NUM_THREADS,
    )
    print(f"[Dataset] Training samples: {len(train_dataset)}  Validation samples: {len(val_dataset)}")

    train_loader = create_loader(
        train_dataset,
        batch_size   = config.BATCH_SIZE,
        shuffle      = True,
        num_workers  = config.LOADER_NUM_WORKERS,
    )
    val_loader = create_loader(
        val_dataset,
        batch_size   = config.BATCH_SIZE,
        shuffle      = False,
        num_workers  = config.LOADER_NUM_WORKERS,
    )
    return train_loader, val_loader
