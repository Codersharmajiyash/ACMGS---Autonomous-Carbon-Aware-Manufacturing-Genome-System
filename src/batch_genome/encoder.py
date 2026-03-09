"""
Phase 3: Batch Genome Encoder

PURPOSE: 
Construct a unified "genome" vector for each production batch by combining:
  - Process parameters (temperature, pressure, speed, feed_rate, humidity)
  - Material properties (density, hardness, grade)
  - Energy DNA embeddings (16-dim latent vectors from Phase 2)
  - Carbon intensity (real-time grid carbon)

WHY:
This genome is the input to prediction models (Phase 4) and optimization (Phase 5).
It's a complete "genetic fingerprint" of manufacturing conditions.

OUTPUT:
- genome_vectors.npy: (num_batches, genome_dim) array
- genome_dim = 5 (process) + 3 (material) + 16 (energy DNA) + 1 (carbon) = 25
"""

import numpy as np
import pandas as pd
import os
from src.utils.logger import get_logger
from config.settings import (
    SIMULATED_DIR, PROCESSED_DIR,
    GENOME_PROCESS_FEATURES, GENOME_MATERIAL_FEATURES
)

logger = get_logger("batch_genome")


def load_batch_data():
    """
    Load the master batch data CSV from Phase 1.
    
    Returns:
        pd.DataFrame: Contains process params, material props, targets, carbon, batch_id
        
    Raises:
        FileNotFoundError: If batch data CSV doesn't exist
        ValueError: If CSV is empty or malformed
    """
    csv_path = os.path.join(SIMULATED_DIR, "batch_data.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"Batch data file not found at: {csv_path}")
        logger.error("Please run Phase 1 first: python -m src.data_simulation.simulator")
        raise FileNotFoundError(f"Batch data not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise ValueError(f"Malformed CSV file: {csv_path}") from e
    
    # Validate data
    if df.empty:
        logger.error("Batch data CSV is empty")
        raise ValueError("Batch data CSV contains no records")
    
    # Check required columns
    required_cols = set(GENOME_PROCESS_FEATURES + GENOME_MATERIAL_FEATURES + 
                       ["carbon_intensity", "batch_id"])
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    logger.info(f"✓ Loaded batch data: {df.shape}")
    logger.info(f"✓ Columns: {list(df.columns)}")
    
    return df


def load_energy_embeddings():
    """
    Load the energy DNA embeddings from Phase 2.
    
    Returns:
        np.ndarray: Shape (num_batches, latent_dim=16)
        
    Raises:
        FileNotFoundError: If embeddings file doesn't exist
        ValueError: If embeddings have wrong shape or contain invalid values
    """
    emb_path = os.path.join(SIMULATED_DIR, "energy_embeddings.npy")
    
    if not os.path.exists(emb_path):
        logger.error(f"Energy embeddings file not found at: {emb_path}")
        logger.error("Please run Phase 2 first: python -m src.energy_dna.trainer")
        raise FileNotFoundError(f"Energy embeddings not found: {emb_path}")
    
    try:
        embeddings = np.load(emb_path)
    except Exception as e:
        logger.error(f"Failed to load embeddings file: {e}")
        raise ValueError(f"Corrupted embeddings file: {emb_path}") from e
    
    # Validate shape
    if embeddings.ndim != 2:
        logger.error(f"Invalid embeddings shape: {embeddings.shape} (expected 2D array)")
        raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
    
    if embeddings.shape[1] != 16:
        logger.error(f"Invalid latent dimension: {embeddings.shape[1]} (expected 16)")
        raise ValueError(f"Expected 16 latent dimensions, got: {embeddings.shape[1]}")
    
    # Check for NaN or Inf values
    if np.any(~np.isfinite(embeddings)):
        logger.error("Embeddings contain NaN or Inf values")
        raise ValueError("Energy embeddings contain invalid values (NaN or Inf)")
    
    logger.info(f"✓ Loaded energy embeddings: {embeddings.shape}")
    
    return embeddings


def extract_process_features(df):
    """
    Extract process parameter features.
    
    Args:
        df (pd.DataFrame): Batch data
        
    Returns:
        np.ndarray: Shape (num_batches, 5)
        
    Raises:
        ValueError: If features contain invalid values
    """
    try:
        features = df[GENOME_PROCESS_FEATURES].values
    except KeyError as e:
        logger.error(f"Missing process feature columns: {e}")
        raise ValueError(f"DataFrame missing process features: {e}") from e
    
    # Validate data
    if np.any(~np.isfinite(features)):
        logger.error("Process features contain NaN or Inf values")
        nan_cols = [GENOME_PROCESS_FEATURES[i] for i in range(features.shape[1]) 
                    if np.any(~np.isfinite(features[:, i]))]
        raise ValueError(f"Invalid values in process features: {nan_cols}")
    
    logger.info(f"✓ Extracted process features: {features.shape}")
    logger.debug(f"Features: {GENOME_PROCESS_FEATURES}")
    
    return features


def extract_material_features(df):
    """
    Extract material property features.
    
    Args:
        df (pd.DataFrame): Batch data
        
    Returns:
        np.ndarray: Shape (num_batches, 3)
        
    Raises:
        ValueError: If features contain invalid values
    """
    try:
        features = df[GENOME_MATERIAL_FEATURES].values
    except KeyError as e:
        logger.error(f"Missing material feature columns: {e}")
        raise ValueError(f"DataFrame missing material features: {e}") from e
    
    # Validate data
    if np.any(~np.isfinite(features)):
        logger.error("Material features contain NaN or Inf values")
        nan_cols = [GENOME_MATERIAL_FEATURES[i] for i in range(features.shape[1]) 
                    if np.any(~np.isfinite(features[:, i]))]
        raise ValueError(f"Invalid values in material features: {nan_cols}")
    
    logger.info(f"✓ Extracted material features: {features.shape}")
    logger.debug(f"Features: {GENOME_MATERIAL_FEATURES}")
    
    return features


def extract_carbon_intensity(df):
    """
    Extract carbon intensity values.
    
    Args:
        df (pd.DataFrame): Batch data
        
    Returns:
        np.ndarray: Shape (num_batches, 1)
        
    Raises:
        ValueError: If carbon values are invalid
    """
    try:
        carbon = df["carbon_intensity"].values.reshape(-1, 1)
    except KeyError:
        logger.error("Missing 'carbon_intensity' column in DataFrame")
        raise ValueError("DataFrame missing 'carbon_intensity' column")
    
    # Validate data
    if np.any(~np.isfinite(carbon)):
        logger.error("Carbon intensity contains NaN or Inf values")
        raise ValueError("Carbon intensity contains invalid values (NaN or Inf)")
    
    # Validate range (typical range: 50-600 gCO2/kWh)
    if np.any(carbon < 0) or np.any(carbon > 1000):
        logger.warning(f"Carbon intensity outside typical range [0-1000]: "
                      f"min={carbon.min():.1f}, max={carbon.max():.1f}")
    
    logger.info(f"✓ Extracted carbon intensity: {carbon.shape}")
    logger.debug(f"Carbon range: [{carbon.min():.1f}, {carbon.max():.1f}] gCO2/kWh")
    
    return carbon


def construct_genome_vectors(df, embeddings):
    """
    Combine all features into a single genome vector for each batch.
    
    Genome structure (25 dimensions):
      [0:5]   - Process parameters (temp, pressure, speed, feed_rate, humidity)
      [5:8]   - Material properties (density, hardness, grade)
      [8:24]  - Energy DNA embeddings (16-dim latent vector)
      [24]    - Carbon intensity
    
    Args:
        df (pd.DataFrame): Batch data
        embeddings (np.ndarray): Energy DNA embeddings
        
    Returns:
        np.ndarray: Genome vectors, shape (num_batches, 25)
    """
    # Extract each component
    process_feats = extract_process_features(df)      # (N, 5)
    material_feats = extract_material_features(df)    # (N, 3)
    carbon_feats = extract_carbon_intensity(df)       # (N, 1)
    
    # Verify shapes match
    num_batches = len(df)
    shapes = {
        'process': process_feats.shape[0],
        'material': material_feats.shape[0],
        'embeddings': embeddings.shape[0],
        'carbon': carbon_feats.shape[0]
    }
    
    mismatches = {k: v for k, v in shapes.items() if v != num_batches}
    if mismatches:
        logger.error(f"Shape mismatch! Expected {num_batches} batches for all features")
        logger.error(f"Mismatched shapes: {mismatches}")
        raise ValueError(f"Feature dimension mismatch: {mismatches}")
    
    logger.debug(f"✓ All feature shapes match: {num_batches} batches")
    
    # Concatenate horizontally
    try:
        genome = np.concatenate([
            process_feats,     # columns 0-4
            material_feats,    # columns 5-7
            embeddings,        # columns 8-23
            carbon_feats       # column 24
        ], axis=1)
    except ValueError as e:
        logger.error(f"Failed to concatenate features: {e}")
        logger.error(f"Shapes: process{process_feats.shape}, material{material_feats.shape}, "
                    f"embeddings{embeddings.shape}, carbon{carbon_feats.shape}")
        raise ValueError("Feature concatenation failed - incompatible shapes") from e
    
    # Validate final genome
    expected_dim = 25  # 5 + 3 + 16 + 1
    if genome.shape[1] != expected_dim:
        logger.error(f"Unexpected genome dimension: {genome.shape[1]} (expected {expected_dim})")
        raise ValueError(f"Genome dimension mismatch: got {genome.shape[1]}, expected {expected_dim}")
    
    logger.info(f"✓ Constructed genome vectors: {genome.shape}")
    logger.info(f"✓ Genome breakdown: Process(5) + Material(3) + EnergyDNA(16) + Carbon(1) = {genome.shape[1]}")
    
    return genome


def normalize_genome(genome):
    """
    Normalize genome vectors using standard scaling (z-score normalization).
    Each feature is scaled to have mean=0 and std=1.
    
    This is crucial for machine learning models to treat all features equally.
    
    Args:
        genome (np.ndarray): Raw genome vectors
        
    Returns:
        tuple: (normalized_genome, mean, std) for inverse transform later
        
    Raises:
        ValueError: If genome contains invalid values
    """
    # Validate input
    if np.any(~np.isfinite(genome)):
        logger.error("Genome contains NaN or Inf values before normalization")
        raise ValueError("Cannot normalize genome with NaN or Inf values")
    
    mean = genome.mean(axis=0)
    std = genome.std(axis=0)
    
    # Check for zero std (constant features)
    zero_std_indices = np.where(std == 0)[0]
    if len(zero_std_indices) > 0:
        logger.warning(f"Features with zero variance (constant): {zero_std_indices}")
        logger.warning("These features will not be normalized")
    
    # Prevent division by zero
    std = np.where(std == 0, 1, std)
    
    normalized = (genome - mean) / std
    
    # Verify normalization
    actual_mean = normalized.mean()
    actual_std = normalized.std()
    if abs(actual_mean) > 1e-6 or abs(actual_std - 1.0) > 0.1:
        logger.warning(f"Normalization check: mean={actual_mean:.6f} (expected ~0), "
                      f"std={actual_std:.6f} (expected ~1)")
    
    logger.info(f"✓ Normalized genome vectors")
    logger.info(f"  Original mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    logger.info(f"  Original std range: [{std.min():.3f}, {std.max():.3f}]")
    logger.info(f"  Normalized mean: {actual_mean:.6f}, std: {actual_std:.6f}")
    
    return normalized, mean, std


def save_genome_data(genome, df, mean=None, std=None):
    """
    Save genome vectors and metadata.
    
    Args:
        genome (np.ndarray): Genome vectors
        df (pd.DataFrame): Original batch data (for batch IDs)
        mean (np.ndarray): Normalization mean (optional)
        std (np.ndarray): Normalization std (optional)
        
    Raises:
        ValueError: If inputs are invalid
        IOError: If file writing fails
    """
    # Validate inputs
    if genome.ndim != 2:
        raise ValueError(f"Genome must be 2D array, got shape: {genome.shape}")
    
    if len(genome) != len(df):
        raise ValueError(f"Genome length ({len(genome)}) doesn't match DataFrame ({len(df)})")
    
    if "batch_id" not in df.columns:
        raise ValueError("DataFrame missing 'batch_id' column")
    
    # Create output directory
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {PROCESSED_DIR}")
        raise IOError(f"Cannot create directory: {PROCESSED_DIR}") from e
    
    # Save genome vectors
    genome_path = os.path.join(PROCESSED_DIR, "genome_vectors.npy")
    try:
        np.save(genome_path, genome)
        logger.info(f"✓ Saved genome vectors → {genome_path}")
    except Exception as e:
        logger.error(f"Failed to save genome vectors: {e}")
        raise IOError(f"Failed to write {genome_path}") from e
    
    # Save normalization parameters if provided
    if mean is not None and std is not None:
        if mean.shape[0] != genome.shape[1] or std.shape[0] != genome.shape[1]:
            raise ValueError(f"Normalization params shape mismatch: mean{mean.shape}, std{std.shape}, genome{genome.shape}")
        
        norm_path = os.path.join(PROCESSED_DIR, "genome_normalization.npz")
        try:
            np.savez(norm_path, mean=mean, std=std)
            logger.info(f"✓ Saved normalization params → {norm_path}")
        except Exception as e:
            logger.error(f"Failed to save normalization params: {e}")
            raise IOError(f"Failed to write {norm_path}") from e
    
    # Save batch IDs for reference
    batch_ids = df["batch_id"].values
    ids_path = os.path.join(PROCESSED_DIR, "batch_ids.npy")
    try:
        np.save(ids_path, batch_ids)
        logger.info(f"✓ Saved batch IDs → {ids_path}")
    except Exception as e:
        logger.error(f"Failed to save batch IDs: {e}")
        raise IOError(f"Failed to write {ids_path}") from e
    
    # Save a metadata CSV for easy inspection
    metadata_df = pd.DataFrame({
        "batch_id": batch_ids,
        "genome_index": np.arange(len(batch_ids))
    })
    metadata_path = os.path.join(PROCESSED_DIR, "genome_metadata.csv")
    try:
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"✓ Saved metadata → {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata CSV: {e}")
        raise IOError(f"Failed to write {metadata_path}") from e


def load_genome_vectors(normalized=True):
    """
    Load saved genome vectors.
    
    Args:
        normalized (bool): If True, return normalized vectors
        
    Returns:
        np.ndarray: Genome vectors
        
    Raises:
        FileNotFoundError: If genome vectors file doesn't exist
        ValueError: If loaded data is invalid
    """
    genome_path = os.path.join(PROCESSED_DIR, "genome_vectors.npy")
    
    if not os.path.exists(genome_path):
        logger.error(f"Genome vectors file not found at: {genome_path}")
        logger.error("Please run the genome pipeline first: python -m src.batch_genome.encoder")
        raise FileNotFoundError(f"Genome vectors not found: {genome_path}")
    
    try:
        genome = np.load(genome_path)
    except Exception as e:
        logger.error(f"Failed to load genome vectors: {e}")
        raise ValueError(f"Corrupted genome file: {genome_path}") from e
    
    # Validate shape
    if genome.ndim != 2 or genome.shape[1] != 25:
        logger.error(f"Invalid genome shape: {genome.shape} (expected (N, 25))")
        raise ValueError(f"Invalid genome shape: {genome.shape}")
    
    logger.info(f"✓ Loaded genome vectors: {genome.shape}")
    
    return genome


def load_normalization_params():
    """
    Load normalization parameters.
    
    Returns:
        tuple: (mean, std)
        
    Raises:
        FileNotFoundError: If normalization file doesn't exist
        ValueError: If normalization params are invalid
    """
    norm_path = os.path.join(PROCESSED_DIR, "genome_normalization.npz")
    
    if not os.path.exists(norm_path):
        logger.error(f"Normalization parameters file not found at: {norm_path}")
        raise FileNotFoundError(f"Normalization params not found: {norm_path}")
    
    try:
        data = np.load(norm_path)
        mean = data["mean"]
        std = data["std"]
    except Exception as e:
        logger.error(f"Failed to load normalization params: {e}")
        raise ValueError(f"Corrupted normalization file: {norm_path}") from e
    
    # Validate shapes
    if mean.shape != (25,) or std.shape != (25,):
        logger.error(f"Invalid normalization params shape: mean{mean.shape}, std{std.shape}")
        raise ValueError(f"Expected shape (25,), got mean{mean.shape}, std{std.shape}")
    
    logger.info(f"✓ Loaded normalization params: mean{mean.shape}, std{std.shape}")
    
    return mean, std


def get_genome_by_batch_id(batch_id):
    """
    Retrieve genome vector for a specific batch.
    
    Args:
        batch_id (str): Batch identifier (e.g., "BATCH_0042")
        
    Returns:
        np.ndarray: Genome vector for that batch
        
    Raises:
        FileNotFoundError: If metadata or genome files don't exist
        ValueError: If batch_id not found
    """
    # Load metadata
    metadata_path = os.path.join(PROCESSED_DIR, "genome_metadata.csv")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found at: {metadata_path}")
        raise FileNotFoundError(f"Genome metadata not found: {metadata_path}")
    
    try:
        metadata = pd.read_csv(metadata_path)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise ValueError(f"Corrupted metadata file: {metadata_path}") from e
    
    # Find index
    row = metadata[metadata["batch_id"] == batch_id]
    if len(row) == 0:
        logger.error(f"Batch ID '{batch_id}' not found in metadata")
        available_ids = metadata["batch_id"].tolist()[:5]
        logger.error(f"Available batch IDs (first 5): {available_ids}")
        raise ValueError(f"Batch ID not found: {batch_id}")
    
    idx = row["genome_index"].values[0]
    
    # Load genome and return specific row
    genome = load_genome_vectors()
    
    if idx >= len(genome):
        logger.error(f"Index {idx} out of range for genome with {len(genome)} batches")
        raise ValueError(f"Invalid genome index: {idx}")
    
    return genome[idx]


def run_batch_genome_pipeline(normalize=True):
    """
    Main pipeline for Phase 3: Construct and save batch genome vectors.
    
    Args:
        normalize (bool): Whether to normalize genome vectors
        
    Returns:
        tuple: (genome, df) for further processing
        
    Raises:
        RuntimeError: If pipeline fails at any stage
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: BATCH GENOME ENCODING PIPELINE STARTING")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load data
        logger.info("Step 1/3: Loading input data...")
        df = load_batch_data()
        embeddings = load_energy_embeddings()
        
        # Step 2: Construct genome
        logger.info("Step 2/3: Constructing genome vectors...")
        genome = construct_genome_vectors(df, embeddings)
        
        # Step 3: Normalize (optional but recommended)
        logger.info("Step 3/3: Normalizing and saving...")
        if normalize:
            genome_normalized, mean, std = normalize_genome(genome)
            save_genome_data(genome_normalized, df, mean, std)
            final_genome = genome_normalized
        else:
            save_genome_data(genome, df)
            final_genome = genome
        
        logger.info("=" * 60)
        logger.info("✓ PHASE 3: COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Total batches processed: {len(df)}")
        logger.info(f"✓ Genome dimension: {final_genome.shape[1]}")
        logger.info(f"✓ Output directory: {PROCESSED_DIR}")
        logger.info(f"✓ Files created:")
        logger.info(f"  - genome_vectors.npy")
        if normalize:
            logger.info(f"  - genome_normalization.npz")
        logger.info(f"  - batch_ids.npy")
        logger.info(f"  - genome_metadata.csv")
        
        return final_genome, df
        
    except FileNotFoundError as e:
        logger.error(f"Pipeline failed - missing input file: {e}")
        logger.error("Please ensure Phase 1 and Phase 2 are complete")
        raise RuntimeError("Phase 3 pipeline failed - missing prerequisite data") from e
    
    except ValueError as e:
        logger.error(f"Pipeline failed - data validation error: {e}")
        raise RuntimeError("Phase 3 pipeline failed - invalid data") from e
    
    except Exception as e:
        logger.error(f"Pipeline failed - unexpected error: {e}")
        raise RuntimeError("Phase 3 pipeline failed") from e


if __name__ == "__main__":
    genome, df = run_batch_genome_pipeline(normalize=True)
    logger.info("Genome shape: %s", genome.shape)
    logger.info("Genome mean (first 5 dims): %s", genome.mean(axis=0)[:5])
    logger.info("Genome std  (first 5 dims): %s", genome.std(axis=0)[:5])
