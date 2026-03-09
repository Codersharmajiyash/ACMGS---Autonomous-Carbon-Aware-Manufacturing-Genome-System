import torch 
import torch.nn as nn 
import numpy as np 
import os 
from torch.utils.data import DataLoader, TensorDataset
from src.energy_dna.model import LSTMAutoencoder
from src.utils.logger import get_logger
from config.settings import(
    SIMULATED_DIR,MODELS_DIR,
    ENERGY_INPUT_DIM,ENERGY_HIDDEN_DIM,ENERGY_LATENT_DIM,ENERGY_NUM_LAYERS,
    ENERGY_EPOCHS,ENERGY_BATCH_SIZE,ENERGY_LEARNING_RATE,
    ENERGY_ANOMALY_PERCENTILE
)
logger = get_logger("energy_dna")


# Load_signals()-> it reads energy signals from npy ile and normalizes  each signal a reshapes for LSTM returns a
# DataLoader and then raw tensors
# RAw  signals  ranges from 50-200 watts.LSTM  gradients become unstable  wuth large values after normalization every signal has mean = 0 and std= 1

def load_signals(npy_paths):
    # Step 1: Load raw numpy array
    signals = np.load(npy_paths)
    # signals shape:(2000,128)

    # Step[ -2 normalize each signal  independentaly]
    # axis = 1 means compute along timespace (columns), not across batches
    mean =  signals.mean(axis=1, keepdims=True)# shape (2000,1)
    std =  signals.std(axis=1, keepdims=True)#shape (2000,1)
    std = np.where(std == 0, 1, std)
    signals = (signals - mean) / std

    #prevention of zero divison when signal is flat 
    signals = signals.reshape(signals.shape[0], signals.shape[1],1)

    # step 4 Convert to Pytorch Tensor 
    # float32 is standard for nueral network training
    tensor = torch.tensor(signals,dtype = torch.float32)

    # Step5: Wrap in DataLoader
    # TensorDataset makes  tensor  iterable  by DataLoader
    dataset = TensorDataset(tensor)

    # DataLoader handles: batching , shuffling, parallel loading
    loader = DataLoader(dataset, batch_size=ENERGY_BATCH_SIZE,shuffle = True)
    logger.info(f"Signals Loaded shape:{signals.shape}")

    return loader, tensor


# Trainmodel creates  model and it also define the loss function  and runs the training loop 

def train_model(loader):
    # Step -1 
    device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
    logger.info(f"Training device:{device}")

    # Step- 2 Create Model and move to device 
    model =  LSTMAutoencoder(
        input_dim = ENERGY_INPUT_DIM,
        hidden_dim=ENERGY_HIDDEN_DIM,
        latent_dim=ENERGY_LATENT_DIM,
        num_layers= ENERGY_NUM_LAYERS
    ).to(device)
    # .to()moves weight to GPU 

    # step-3  Loss Function"
    # MSELOSS= mean sqaure loss = average of(actual - predicted)power 2
    # PERFECT FOR AUTOENCODERS: It will measure reconstrcution quality
    criterion = nn.MSELoss()

    # Step-4 Optimizer 
    # ADAM: ADAPTIVE LEARNING RATE OPTIMIZER ->  
    optimizer = torch.optim.Adam(model.parameters(),lr=ENERGY_LEARNING_RATE)

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # STEP-5 : Traininf Loop 
    model.train() # set to trainin mode (enbles dropout  etc. if any)

    best_loss = float('inf')
    for epoch in range(ENERGY_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for(batch,) in loader:#Dataloder will return tuple unpack with the batch
            batch= batch.to(device)
            # Forward pass
            reconstructed,_= model(batch)
            #reconstructed shape: same as batch(batch_size,128,1)
            # _ is the latent - not needed during training
            loss = criterion(reconstructed, batch)
            # compare reconstruction vs original training
            # Backward pass
            optimizer.zero_grad()#Must clear old gradients first
            loss.backward()#compute gradients
            optimizer.step()#update all weights

            epoch_loss += loss.item() # .item()converts tensor -> float
            num_batches +=1
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if(epoch +1 ) % 10 == 0:
            logger.info(f"Epoch[{epoch+1:3d}/{ENERGY_EPOCHS}] | Avg Loss: {avg_loss:.6f}")
    logger.info("Training Complete")
    return model

def save_model(model):
    os.makedirs(MODELS_DIR,exist_ok=True)

    save_path = os.path.join(MODELS_DIR,"lstm_autoencoder.pth")
    # state_dict = dictionary of all leaninf parameter tensors
    torch.save(model.state_dict(),save_path)

    logger.info(f"Model Saved->{save_path}")
    return save_path

def extract_embeddings(model,tensor):
    device = torch.device("cuda"if torch.cuda.is_available()else "cpu")

    # SWITCH TO EVALUATE : ALWAYS DISABLE DROPOUT ,ALWAYS DISABLE BATCH NORM UPDATES ALWAYS DO BEFORE INFERENCE
    model.eval()

    with torch.no_grad():
        # torch.no_grad() = dont track gradients
        # During inference you dont need them saves GPU + speeds up
        tensor = tensor.to(device)
        # Runs a full forward pass we want latent not reconstruction
        _, embeddings = model(tensor)

        # Move to CPU memory Numpy doesnt work on CUDA tensors
        embeddings = embeddings.cpu().numpy()

        # Save embeddings for Phase 3
        os.makedirs(SIMULATED_DIR,exist_ok=True)
        emb_path = os.path.join(SIMULATED_DIR, "energy_embeddings.npy")
    np.save(emb_path, embeddings)

    logger.info(f"Embeddings extracted shape: {embeddings.shape}")
    logger.info(f"Embeddings saved → {emb_path}")

    return embeddings

def detect_anomalies(model,tensor):
    device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
    model.eval()

    with torch.no_grad():
        tensor = tensor.to(device)
        reconstructed,_ = model(tensor)
        # Compute MSE per signal  
        # (tensor- reconstructed)power 2 shape:(2000,128,1) averages across timesteps and features
        # .mean(dim=[1,2])result shape (2000,)  one error value per signal
        # 
        errors=((tensor-reconstructed)**2).mean(dim=[1,2]).cpu().numpy()

        # Threshhold: any signal  above 95th percentile of errors = anomaly
        threshold = np.percentile(errors,ENERGY_ANOMALY_PERCENTILE)

        # Boolean array: True where error exceeds threshold 
        anomalies = errors > threshold

        logger.info(f"Reconstruction error - min:{errors.min():.6f}|max: {errors.max():.6f}")
        logger.info(f"Anomaly threshold(p{ENERGY_ANOMALY_PERCENTILE}): {threshold:.6f}")
        logger.info(f"Anomalies flagged:{anomalies.sum()}/ {len(anomalies)}")

        return errors,anomalies 
    

def run_energy_dna_pipeline():
    # MAIN PIPLINE  LOAD -> TRAIN -> SAVE -> EXTRACT EMBEDDINGS -> DETECT ANOMALIES

    npy_path = os.path.join(SIMULATED_DIR,"energy_signals.npy")

    logger.info("="*50)
    logger.info("ENERGY DNA PIPLINE STARTING")
    logger.info("="*50)

    loader, tensor = load_signals(npy_path)
    model = train_model(loader)
    save_model(model)
    embeddings = extract_embeddings(model,tensor)
    errors, flags = detect_anomalies(model,tensor)

    logger.info("PHASE 2:COMPLETE")

    return embeddings, errors, flags


if __name__ == "__main__":
    embeddings, errors, flags = run_energy_dna_pipeline()

    print(f"\n{'='*40}")
    print(f"Embeddings shape : {embeddings.shape}")   # (2000, 16)
    print(f"Total anomalies  : {flags.sum()}")         # ~100
    print(f"Error range      : {errors.min():.4f} → {errors.max():.4f}")
    print(f"{'='*40}")


        