import os


# ============================================================
# Global configuration
# ============================================================

# --- CPU thread control: these variables are set before importing torch in main.py ---
OMP_NUM_THREADS = "1"  # Number of OpenMP threads; increase it on servers if needed
MKL_NUM_THREADS = "1"  # Number of MKL threads; increase it on servers if needed

# --- Training parameters ---
EPOCHS = 1000  # Maximum number of training epochs
LR = 1e-5  # Learning rate
BATCH_SIZE = 16  # Number of samples per batch
SEED = 42  # Random seed for reproducibility
LOAD_TRAINED_WEIGHTS = False  # Whether to continue training from an existing checkpoint

# --- Input mode ---
USE_FIRST_STEP_INPUT = True  # True: use the first-step spectrum as input; False: use physical parameters parsed from filenames

# --- Data and output paths ---
TRAIN_DATA_FOLDER = r"data/spectral/train"  # Training dataset directory
VAL_DATA_FOLDER = r"data/spectral/val"  # Validation dataset directory
OUTPUT_DIR = "result"  # Directory for checkpoints and prediction results

# --- Data shape ---
FIBER_STEPS = 200  # Number of propagation steps; automatically updated from .mat shape at runtime
OUTPUT_SIZE = 251  # Spectrum length; automatically updated from .mat shape at runtime
INPUT_SIZE = OUTPUT_SIZE if USE_FIRST_STEP_INPUT else 3  # Model input dimension

# --- Model architecture ---
CHANNEL_LIST = [256, 256, 256]  # Encoder channel sizes
KERNEL_PARAMS = [(1, 0), (3, 1), (5, 2)]  # Encoder kernel sizes and padding values
EXPERT_DIM = 256  # Internal channel size for expert networks
GLOBAL_COND_DIM = 256  # Global condition vector dimension
NUM_EXPERTS = 4  # Number of experts in each MoE stage
TOP_K = 2  # Number of activated experts for each sample
MOE_AUX_WEIGHT = 0.01  # Weight of the expert load-balancing auxiliary loss

# --- Data loading parameters ---
PRELOAD_NUM_THREADS = min(16, max(4, os.cpu_count() or 8))  # Number of threads for preloading .mat files
LOADER_NUM_WORKERS = 0 if os.name == "nt" else min(4, max(1, (os.cpu_count() or 4) // 2))  # Number of DataLoader workers
PREFETCH_FACTOR = 4  # Number of prefetched batches for multiprocessing DataLoader

# --- Output files ---
RUN_NAME = f"MambaMoE_{'firstStep' if USE_FIRST_STEP_INPUT else 'params'}_top{TOP_K}"  # Run name
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}.pth")  # Best checkpoint save path
RESULT_SAVE_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_predictions.mat")  # Train and validation prediction result path
