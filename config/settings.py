"""
ACMGS Configuration — Central settings for all modules.

WHY: A single config file prevents magic numbers scattered across code.
Every module imports from here, so changing a value here changes it everywhere.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SIMULATED_DIR = os.path.join(DATA_DIR, "simulated")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "acmgs.db")

# ─── Data Simulation (Phase 1) ───────────────────────────────────
SIM_NUM_MACHINES = 10            # Number of machines to simulate
SIM_NUM_BATCHES = 2000           # Number of production batches
SIM_SIGNAL_LENGTH = 128          # Time-steps per energy signal
SIM_SAMPLING_RATE_HZ = 10       # Simulated sampling rate
SIM_RANDOM_SEED = 42            # Reproducibility

# ─── Energy DNA Model (Phase 2) ──────────────────────────────────
ENERGY_INPUT_DIM = 1             # Univariate power signal
ENERGY_HIDDEN_DIM = 64           # LSTM hidden units
ENERGY_LATENT_DIM = 16           # Embedding size (the "DNA")
ENERGY_NUM_LAYERS = 2            # LSTM layers
ENERGY_EPOCHS = 50               # Training epochs
ENERGY_BATCH_SIZE = 64           # Training batch size
ENERGY_LEARNING_RATE = 1e-3
ENERGY_ANOMALY_PERCENTILE = 95   # Reconstruction error threshold

# ─── Batch Genome (Phase 3) ──────────────────────────────────────
GENOME_PROCESS_FEATURES = [      # Process parameter column names
    "temperature", "pressure", "speed", "feed_rate", "humidity"
]
GENOME_MATERIAL_FEATURES = [     # Material profile columns
    "material_density", "material_hardness", "material_grade"
]

# ─── Prediction Model (Phase 4) ──────────────────────────────────
PRED_TARGETS = ["yield", "quality", "energy_consumption"]
PRED_TEST_SIZE = 0.2
PRED_RANDOM_STATE = 42
PRED_N_ESTIMATORS = 200
PRED_MAX_DEPTH = 6

# ─── Optimization Engine (Phase 5) ───────────────────────────────
OPT_POPULATION_SIZE = 100
OPT_NUM_GENERATIONS = 50
OPT_CROSSOVER_PROB = 0.8
OPT_MUTATION_PROB = 0.2
OPT_NUM_OBJECTIVES = 4          # yield, quality, energy, carbon

# ─── Carbon Scheduler (Phase 6) ──────────────────────────────────
CARBON_HIGH_THRESHOLD = 400      # gCO2/kWh — above = high carbon
CARBON_LOW_THRESHOLD = 150       # gCO2/kWh — below = low carbon

# ─── API (Phase 8) ───────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Dashboard (Phase 9) ─────────────────────────────────────────
DASH_PORT = 8501
