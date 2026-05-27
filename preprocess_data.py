import argparse
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio


RAW_DATASETS = {
    "spectral": {
        "raw_path": Path("data/SC_spec_251.mat"),  # Raw spectral-domain data file
        "output_dir": Path("data/spectral"),  # Preprocessed spectral-domain output directory
        "normalization": "dBm",  # Spectral data is represented on a clipped logarithmic scale
    },
    "time": {
        "raw_path": Path("data/SC_time_276.mat"),  # Raw time-domain data file
        "output_dir": Path("data/time"),  # Preprocessed time-domain output directory
        "normalization": "max",  # Temporal data remains on a linear intensity scale
    },
}


def normalize_spectral_dBm(data: np.ndarray, dynamic_range_db: float) -> tuple[np.ndarray, dict]:
    """Normalize spectral data with global max scaling, log transform, clipping, and 0-1 mapping."""
    max_value = float(np.max(np.abs(data)))
    if max_value <= 0:
        raise ValueError("The maximum value of the spectral data is zero; dBm normalization cannot be performed.")

    normalized = data / max_value
    with np.errstate(divide="ignore", invalid="ignore"):
        db_data = 10.0 * np.log10(normalized)

    db_data = np.nan_to_num(db_data, nan=-dynamic_range_db, neginf=-dynamic_range_db)
    db_data[db_data < -dynamic_range_db] = -dynamic_range_db
    db_data = db_data / dynamic_range_db + 1.0
    db_data = np.clip(db_data, 0.0, 1.0).astype(np.float32)
    metadata = {
        "normalization": "dBm",
        "max_value": max_value,
        "dynamic_range_db": dynamic_range_db,
    }
    return db_data, metadata


def normalize_temporal_linear(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize temporal intensity data linearly with a global maximum absolute value."""
    max_value = float(np.max(np.abs(data)))
    if max_value <= 0:
        raise ValueError("The maximum value of the temporal data is zero; linear normalization cannot be performed.")

    normalized = np.clip(data / max_value, 0.0, 1.0).astype(np.float32)
    metadata = {
        "normalization": "max",
        "max_value": max_value,
        "dynamic_range_db": np.nan,
    }
    return normalized, metadata


def normalize_dataset(
    dataset_name: str,
    data: np.ndarray,
    normalization: str,
    dynamic_range_db: float,
) -> tuple[np.ndarray, dict]:
    """Apply the domain-specific normalization selected for a raw dataset."""
    if normalization == "dBm":
        normalized_data, metadata = normalize_spectral_dBm(data, dynamic_range_db)
    elif normalization == "max":
        normalized_data, metadata = normalize_temporal_linear(data)
    else:
        raise ValueError(f"Unsupported normalization for {dataset_name}: {normalization}")

    metadata["dataset_name"] = dataset_name
    return normalized_data, metadata


def prepare_output_dir(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    """Create train and validation output directories."""
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.rglob("*.mat")):
        raise FileExistsError(
            f".mat files already exist in {output_dir}. Use --overwrite to regenerate."
        )

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir


def save_split_samples(
    data: np.ndarray,
    train_dir: Path,
    val_dir: Path,
    train_count: int,
) -> None:
    """Split a 3D raw array into per-sample .mat files with the field name new_lIW."""
    sample_count = data.shape[0]
    for index in range(sample_count):
        sample = np.ascontiguousarray(data[index].T)
        split_dir = train_dir if index < train_count else val_dir
        sample_name = f"{index + 1:04d}.mat"
        sio.savemat(split_dir / sample_name, {"new_lIW": sample}, do_compression=False)

        if (index + 1) % 100 == 0 or index + 1 == sample_count:
            print(f"[Progress] Saved {index + 1}/{sample_count} samples")


def preprocess_one_dataset(
    dataset_name: str,
    raw_path: Path,
    output_dir: Path,
    normalization: str,
    train_ratio: float,
    dynamic_range_db: float,
    overwrite: bool,
) -> None:
    """Preprocess one raw data file."""
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    print(f"\n[Data] Loading {dataset_name}: {raw_path}")
    mat_data = sio.loadmat(raw_path)
    if "data" not in mat_data:
        raise KeyError(f"Field 'data' not found in {raw_path}")

    raw_data = np.asarray(mat_data["data"], dtype=np.float64)
    if raw_data.ndim != 3:
        raise ValueError(f"Data in {raw_path} should be a 3D array; actual shape: {raw_data.shape}")

    print(f"[Data] Original shape: {raw_data.shape}")
    normalized_data, norm_metadata = normalize_dataset(
        dataset_name,
        raw_data,
        normalization,
        dynamic_range_db,
    )
    print(f"[Normalization] Method: {norm_metadata['normalization']}")
    print(f"[Normalization] Global maximum value: {norm_metadata['max_value']:.10g}")
    if norm_metadata["normalization"] == "dBm":
        print(f"[Normalization] Dynamic range: {dynamic_range_db:g} dB")
    print(
        f"[Normalization] Normalized range: "
        f"{float(normalized_data.min()):.6f} ~ {float(normalized_data.max()):.6f}"
    )

    sample_count = int(normalized_data.shape[0])
    train_count = int(round(sample_count * train_ratio))
    train_count = min(max(train_count, 1), sample_count - 1)

    train_dir, val_dir = prepare_output_dir(output_dir, overwrite)
    save_split_samples(normalized_data, train_dir, val_dir, train_count)

    metadata = {
        "source_file": str(raw_path),
        "dataset_name": dataset_name,
        "normalization": norm_metadata["normalization"],
        "max_value": norm_metadata["max_value"],
        "dynamic_range_db": norm_metadata["dynamic_range_db"],
        "sample_count": sample_count,
        "train_count": train_count,
        "val_count": sample_count - train_count,
        "sample_shape": np.array([[normalized_data.shape[2], normalized_data.shape[1]]]),
    }
    sio.savemat(output_dir / "normalization_info.mat", metadata, do_compression=False)
    print(
        f"[Done] {dataset_name}: {train_count} training samples, "
        f"{sample_count - train_count} validation samples, output directory: {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw supercontinuum .mat data into the format used by this training code"
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "spectral", "time"],
        default="all",
        help="Dataset to preprocess. Use all to process both spectral-domain and time-domain data.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio. Default: 0.8.",
    )
    parser.add_argument(
        "--dynamic-range-db",
        type=float,
        default=55.0,
        help="Dynamic range for dBm normalization. Default: 55 dB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing preprocessed .mat files before regenerating the output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.dynamic_range_db <= 0:
        raise ValueError("--dynamic-range-db must be greater than 0.")

    names = ["spectral", "time"] if args.dataset == "all" else [args.dataset]
    for name in names:
        item = RAW_DATASETS[name]
        preprocess_one_dataset(
            dataset_name=name,
            raw_path=item["raw_path"],
            output_dir=item["output_dir"],
            normalization=item["normalization"],
            train_ratio=args.train_ratio,
            dynamic_range_db=args.dynamic_range_db,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
