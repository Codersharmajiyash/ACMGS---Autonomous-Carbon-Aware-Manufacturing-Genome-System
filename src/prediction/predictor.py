"""
Phase 4: Multi-Target Prediction Models

PURPOSE:
Train machine learning models to predict manufacturing outcomes from genome vectors:
  - Yield (batch output ratio: 0.5-1.0)
  - Quality (product quality score: 0.3-1.0)
  - Energy Consumption (kWh: 50-500)

WHY:
These predictions enable:
  - Rapid "what-if" analysis for optimization (Phase 5)
  - Production planning without running full simulations
  - Real-time decision support during manufacturing

APPROACH:
  - Multi-target regression (predict all 3 targets simultaneously)
  - Ensemble methods (XGBoost for accuracy + speed)
  - Train/test split for proper validation
  - Comprehensive metrics (MAE, RMSE, R²)

INPUT:
  - X: Genome vectors from Phase 3 (2000 × 25)
  - y: Target values from Phase 1 (2000 × 3)

OUTPUT:
  - Trained models saved to models/saved/
  - Performance metrics logged
  - Prediction utilities for Phase 5
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from src.utils.logger import get_logger
from config.settings import (
    SIMULATED_DIR, PROCESSED_DIR, MODELS_DIR,
    PRED_TARGETS, PRED_TEST_SIZE, PRED_RANDOM_STATE,
    PRED_N_ESTIMATORS, PRED_MAX_DEPTH
)

logger = get_logger("prediction")

# Try to import XGBoost (fallback to RandomForest if not available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost available - will use for prediction models")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - falling back to RandomForest")


def load_genome_features():
    """
    Load genome vectors (features) from Phase 3.
    
    Returns:
        np.ndarray: Genome vectors, shape (num_batches, 25)
        
    Raises:
        FileNotFoundError: If genome vectors don't exist
        ValueError: If data is invalid
    """
    genome_path = os.path.join(PROCESSED_DIR, "genome_vectors.npy")
    
    if not os.path.exists(genome_path):
        logger.error(f"Genome vectors not found at: {genome_path}")
        logger.error("Please run Phase 3 first: python -m src.batch_genome.encoder")
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
    
    # Check for invalid values
    if np.any(~np.isfinite(genome)):
        logger.error("Genome contains NaN or Inf values")
        raise ValueError("Genome vectors contain invalid values")
    
    logger.info(f"✓ Loaded genome features: {genome.shape}")
    
    return genome


def load_target_values():
    """
    Load target values (yield, quality, energy) from Phase 1 batch data.
    
    Returns:
        tuple: (targets_array, target_names)
            - targets_array: np.ndarray, shape (num_batches, 3)
            - target_names: list of target column names
    
    Raises:
        FileNotFoundError: If batch data doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    csv_path = os.path.join(SIMULATED_DIR, "batch_data.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"Batch data not found at: {csv_path}")
        logger.error("Please run Phase 1 first: python -m src.data_simulation.simulator")
        raise FileNotFoundError(f"Batch data not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read batch data: {e}")
        raise ValueError(f"Corrupted CSV file: {csv_path}") from e
    
    # Check required columns exist
    missing_cols = set(PRED_TARGETS) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing target columns: {missing_cols}")
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"CSV missing target columns: {missing_cols}")
    
    # Extract target values
    try:
        targets = df[PRED_TARGETS].values
    except KeyError as e:
        logger.error(f"Failed to extract target columns: {e}")
        raise ValueError(f"Cannot extract targets: {e}") from e
    
    # Validate data
    if np.any(~np.isfinite(targets)):
        logger.error("Target values contain NaN or Inf")
        nan_cols = [PRED_TARGETS[i] for i in range(len(PRED_TARGETS)) 
                    if np.any(~np.isfinite(targets[:, i]))]
        raise ValueError(f"Invalid values in target columns: {nan_cols}")
    
    logger.info(f"✓ Loaded target values: {targets.shape}")
    logger.info(f"✓ Target columns: {PRED_TARGETS}")
    logger.debug(f"  Yield range: [{targets[:, 0].min():.3f}, {targets[:, 0].max():.3f}]")
    logger.debug(f"  Quality range: [{targets[:, 1].min():.3f}, {targets[:, 1].max():.3f}]")
    logger.debug(f"  Energy range: [{targets[:, 2].min():.1f}, {targets[:, 2].max():.1f}]")
    
    return targets, PRED_TARGETS


def split_train_test(X, y, test_size=PRED_TEST_SIZE, random_state=PRED_RANDOM_STATE):
    """
    Split data into training and testing sets.
    
    Args:
        X (np.ndarray): Features (genome vectors)
        y (np.ndarray): Targets (yield, quality, energy)
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        
    Raises:
        ValueError: If inputs have mismatched shapes
    """
    # Validate shapes match
    if len(X) != len(y):
        logger.error(f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples")
        raise ValueError(f"X and y must have same number of samples: {len(X)} != {len(y)}")
    
    # Check test_size is valid
    if not 0 < test_size < 1:
        logger.error(f"Invalid test_size: {test_size} (must be between 0 and 1)")
        raise ValueError(f"test_size must be between 0 and 1, got: {test_size}")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
    except Exception as e:
        logger.error(f"Train/test split failed: {e}")
        raise ValueError("Failed to split data") from e
    
    logger.info(f"✓ Data split complete:")
    logger.info(f"  Training set: {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    logger.info(f"  Test set:     {X_test.shape[0]} samples ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def create_predictor_model(use_xgboost=None):
    """
    Create a multi-target regression model.
    
    Args:
        use_xgboost (bool): If True, use XGBoost; if False, use RandomForest.
                           If None, auto-detect based on availability.
    
    Returns:
        MultiOutputRegressor: Model capable of predicting multiple targets
        
    Raises:
        ValueError: If XGBoost requested but not available
    """
    # Auto-detect if not specified
    if use_xgboost is None:
        use_xgboost = XGBOOST_AVAILABLE
    
    # Validate XGBoost availability
    if use_xgboost and not XGBOOST_AVAILABLE:
        logger.error("XGBoost requested but not installed")
        logger.error("Install with: pip install xgboost")
        raise ValueError("XGBoost not available. Install or set use_xgboost=False")
    
    if use_xgboost:
        # XGBoost model (faster, often more accurate)
        base_model = xgb.XGBRegressor(
            n_estimators=PRED_N_ESTIMATORS,
            max_depth=PRED_MAX_DEPTH,
            learning_rate=0.1,
            random_state=PRED_RANDOM_STATE,
            n_jobs=-1  # Use all CPU cores
        )
        model_type = "XGBoost"
    else:
        # RandomForest model (robust fallback)
        base_model = RandomForestRegressor(
            n_estimators=PRED_N_ESTIMATORS,
            max_depth=PRED_MAX_DEPTH,
            random_state=PRED_RANDOM_STATE,
            n_jobs=-1
        )
        model_type = "RandomForest"
    
    # Wrap in MultiOutputRegressor for multi-target prediction
    model = MultiOutputRegressor(base_model)
    
    logger.info(f"✓ Created {model_type} multi-target predictor")
    logger.info(f"  n_estimators: {PRED_N_ESTIMATORS}")
    logger.info(f"  max_depth: {PRED_MAX_DEPTH}")
    
    return model


def train_model(model, X_train, y_train):
    """
    Train the prediction model.
    
    Args:
        model: Scikit-learn compatible model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        
    Returns:
        model: Trained model
        
    Raises:
        ValueError: If training fails
    """
    logger.info("Training model...")
    
    # Validate inputs
    if X_train.shape[0] == 0:
        logger.error("Training set is empty")
        raise ValueError("Cannot train on empty dataset")
    
    if X_train.shape[0] != y_train.shape[0]:
        logger.error(f"Training data mismatch: X has {X_train.shape[0]}, y has {y_train.shape[0]}")
        raise ValueError("X_train and y_train must have same number of samples")
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise ValueError("Model training failed") from e
    
    logger.info("✓ Model training complete")
    
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): True test targets
        target_names (list): Names of target variables
        
    Returns:
        dict: Performance metrics for each target
        
    Raises:
        ValueError: If evaluation fails
    """
    logger.info("Evaluating model on test set...")
    
    # Make predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError("Model prediction failed") from e
    
    # Validate prediction shape
    if y_pred.shape != y_test.shape:
        logger.error(f"Prediction shape mismatch: {y_pred.shape} != {y_test.shape}")
        raise ValueError("Prediction shape doesn't match test targets")
    
    # Calculate metrics for each target
    metrics = {}
    
    for i, target_name in enumerate(target_names):
        y_true_i = y_test[:, i]
        y_pred_i = y_pred[:, i]
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_i, y_pred_i)
        rmse = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
        r2 = r2_score(y_true_i, y_pred_i)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mape = np.mean(np.abs((y_true_i - y_pred_i) / np.clip(y_true_i, 1e-8, None))) * 100
        
        metrics[target_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }
        
        logger.info(f"✓ {target_name}:")
        logger.info(f"    MAE:  {mae:.4f}")
        logger.info(f"    RMSE: {rmse:.4f}")
        logger.info(f"    R²:   {r2:.4f}")
        logger.info(f"    MAPE: {mape:.2f}%")
    
    # Overall metrics
    overall_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    logger.info(f"✓ Overall R² Score: {overall_r2:.4f}")
    
    metrics["overall"] = {"R2": overall_r2}
    
    return metrics


def save_model(model, metrics, model_name="predictor"):
    """
    Save trained model and its performance metrics.
    
    Args:
        model: Trained model
        metrics (dict): Performance metrics
        model_name (str): Base name for saved files
        
    Returns:
        tuple: (model_path, metrics_path)
        
    Raises:
        IOError: If saving fails
    """
    # Create models directory
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create models directory: {MODELS_DIR}")
        raise IOError(f"Cannot create directory: {MODELS_DIR}") from e
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Saved model → {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise IOError(f"Failed to write {model_path}") from e
    
    # Save metrics as JSON
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.pkl")
    try:
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        logger.info(f"✓ Saved metrics → {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise IOError(f"Failed to write {metrics_path}") from e
    
    return model_path, metrics_path


def load_model(model_name="predictor"):
    """
    Load a saved model.
    
    Args:
        model_name (str): Base name of the model file
        
    Returns:
        model: Loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is corrupted
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        logger.error("Please train the model first")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ValueError(f"Corrupted model file: {model_path}") from e
    
    logger.info(f"✓ Loaded model from {model_path}")
    
    return model


def predict(model, genome_vectors):
    """
    Make predictions for new genome vectors.
    
    Args:
        model: Trained model
        genome_vectors (np.ndarray): Genome vectors to predict for, shape (N, 25)
        
    Returns:
        np.ndarray: Predictions, shape (N, 3) [yield, quality, energy]
        
    Raises:
        ValueError: If input shape is invalid
    """
    # Validate input
    if genome_vectors.ndim == 1:
        # Single sample - reshape to 2D
        genome_vectors = genome_vectors.reshape(1, -1)
    
    if genome_vectors.ndim != 2 or genome_vectors.shape[1] != 25:
        logger.error(f"Invalid genome shape: {genome_vectors.shape} (expected (N, 25))")
        raise ValueError(f"Genome vectors must have 25 dimensions, got shape: {genome_vectors.shape}")
    
    if np.any(~np.isfinite(genome_vectors)):
        logger.error("Input contains NaN or Inf values")
        raise ValueError("Genome vectors contain invalid values")
    
    # Make predictions
    try:
        predictions = model.predict(genome_vectors)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError("Prediction failed") from e
    
    return predictions


def run_prediction_pipeline(use_xgboost=None, save_results=True):
    """
    Main pipeline for Phase 4: Train and evaluate prediction models.
    
    Args:
        use_xgboost (bool): Whether to use XGBoost (None = auto-detect)
        save_results (bool): Whether to save trained model
        
    Returns:
        tuple: (model, metrics, X_test, y_test, y_pred)
        
    Raises:
        RuntimeError: If pipeline fails at any stage
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: PREDICTION MODEL PIPELINE STARTING")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load data
        logger.info("Step 1/5: Loading data...")
        X = load_genome_features()
        y, target_names = load_target_values()
        
        # Step 2: Split data
        logger.info("Step 2/5: Splitting train/test sets...")
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        
        # Step 3: Create and train model
        logger.info("Step 3/5: Creating and training model...")
        model = create_predictor_model(use_xgboost=use_xgboost)
        model = train_model(model, X_train, y_train)
        
        # Step 4: Evaluate
        logger.info("Step 4/5: Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test, target_names)
        
        # Step 5: Save
        if save_results:
            logger.info("Step 5/5: Saving model and metrics...")
            save_model(model, metrics)
        else:
            logger.info("Step 5/5: Skipping save (save_results=False)")
        
        # Make predictions on test set for return
        y_pred = model.predict(X_test)
        
        logger.info("=" * 60)
        logger.info("✓ PHASE 4: COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Model trained on {X_train.shape[0]} samples")
        logger.info(f"✓ Model tested on {X_test.shape[0]} samples")
        logger.info(f"✓ Overall R² Score: {metrics['overall']['R2']:.4f}")
        logger.info(f"✓ Model saved to: {MODELS_DIR}")
        
        return model, metrics, X_test, y_test, y_pred
        
    except FileNotFoundError as e:
        logger.error(f"Pipeline failed - missing input file: {e}")
        logger.error("Please ensure Phase 1 and Phase 3 are complete")
        raise RuntimeError("Phase 4 pipeline failed - missing prerequisite data") from e
    
    except ValueError as e:
        logger.error(f"Pipeline failed - data validation error: {e}")
        raise RuntimeError("Phase 4 pipeline failed - invalid data") from e
    
    except Exception as e:
        logger.error(f"Pipeline failed - unexpected error: {e}")
        raise RuntimeError("Phase 4 pipeline failed") from e


if __name__ == "__main__":
    # Run the prediction pipeline
    model, metrics, X_test, y_test, y_pred = run_prediction_pipeline()
    
    print(f"\n{'='*50}")
    print("PREDICTION MODEL RESULTS:")
    print(f"{'='*50}")
    print(f"Model type: {'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest'}")
    print(f"Training samples: {len(y_test) * (1-PRED_TEST_SIZE) / PRED_TEST_SIZE:.0f}")
    print(f"Test samples: {len(y_test)}")
    print(f"\nPer-Target Performance:")
    for target in PRED_TARGETS:
        print(f"\n{target.upper()}:")
        print(f"  MAE:  {metrics[target]['MAE']:.4f}")
        print(f"  RMSE: {metrics[target]['RMSE']:.4f}")
        print(f"  R²:   {metrics[target]['R2']:.4f}")
    print(f"\nOverall R² Score: {metrics['overall']['R2']:.4f}")
    print(f"{'='*50}")
