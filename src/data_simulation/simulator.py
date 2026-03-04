import numpy as np 
import pandas as pd 
import os 

from config.settings import (SIMULATED_DIR, SIM_NUM_BATCHES, SIM_SIGNAL_LENGTH, SIM_RANDOM_SEED, SIM_NUM_MACHINES)

from src.utils.logger import get_logger

logger = get_logger("data simulation")
def generate_energy_signals(num_batches,signal_length,seed):
    np.random.seed(seed)
    signals = []

    for i in range(num_batches):
        #Time axis: 128 evenly spaced points from 0 to 2pie(TO GET A FULL WAVE)
        t = np.linspace(0,2*np.pi,signal_length)
        
        #EACH MACHINE  HAS DIFFRENT OPERATING PROPERTIES 
        freq = np.random.uniform(1,5)#generates a randome frequency (oscillation speed)
        amp = np.random.uniform(50,100)#generates a random amplitude (power levels in watts)
        #Base Signal: a smooth sine wave representing  normal machine power draw
        signal = amp *np.sin(freq*t)
        # Add small Gaussian noise (real sensors are never perfectly clean) so we have to add noise in this
        signal += np.random.normal(0,amp*0.1,signal_length)
        #20% of batches : simulate  machine degradation
        if np.random.random() < 0.2:
            #Heav noise: std dev = 30 % of amplitude erractic power draw
            signal += np.random.normal(0,amp * 0.3,signal_length)
            #Linear drift: power gradaully increases (overheating/wear)
            signal += np.linspace(0,amp*0.2,signal_length)
        signals.append(signal)
        # convert list of arrays  single 2D nupy array 

    return np.array(signals)

def generate_process_parameters(num_batches,seed):
#This function will generate the  readings for each batch 
#Retunr the  :DataFrames with the shape (num_batches,5)
    np.random.seed(seed)

    data = {
        "temperature": np.random.uniform(150 , 350 , num_batches),
        "pressure": np.random.uniform(1.0 , 10.0, num_batches),
        "speed": np.random.uniform( 500, 3000, num_batches),
        "feed_rate": np.random.uniform( 0.1, 2.0, num_batches),
        "humidity": np.random.uniform( 20, 80, num_batches),
    }
    return pd.DataFrame(data)


def generate_material_profiles(num_batches, seed):
    # Generate material property profile for each batch
    #Returns DataFrame with Shape(num_batches)

    np.random.seed(seed)

    data = {
        "material_density": np.random.uniform(1.0,8.0, num_batches),
        "material_hardness": np.random.uniform(20, 90, num_batches),
        "material_grade": np.random.choice([1,2,3], size=num_batches),
    }
    return pd.DataFrame(data)

def generate_targets(process_df,material_df,seed):

    np.random.seed(seed)
    n = len(process_df)

    # Yield Formula
    yield_vals =(
        0.5
        + 0.001 * process_df["temperature"].values
        + 0.02 * process_df["pressure"].values
        - 0.00008 * process_df["speed"].values
        + 0.05 * material_df["material_grade"].values
        + np.random.normal(0,0.03,n)#small noise
    )
    yield_vals = np.clip(yield_vals,0.5,1.0)

    # Quality Formula
    quality_vals = (
        0.3
        + 0.005 * material_df["material_hardness"].values
        - 0.00015 * process_df["speed"].values
        + 0.08 * material_df["material_grade"].values
        + np.random.normal(0,0.04,n)#Larger noise for energy
    )
    quality_vals = np.clip(quality_vals,0.3,1.0)

    energy_vals = (
        50
        + 0.1 * process_df["speed"].values
        + 20 * process_df["pressure"].values
        + 80 * process_df["feed_rate"].values
        + 10 * material_df["material_density"].values
        + np.random.normal(0,15,n)#Larger noise for energy
    )
    energy_vals = np.clip(energy_vals,50,500)
    return pd.DataFrame({
        "yield": yield_vals,
        "quality": quality_vals,
        "energy_consumption": energy_vals,
    })

def generate_carbon_intensity(num_batches,seed):
    #Simulating Carbon emition   instensity for each batch
    np.random.seed(seed)
    #Assuming the Batches
    batches_per_day = 24
    #Time axis: one full sine cycle = one day
   
    t = np.arange(num_batches)
    #Base Carbon level
    base = 300
    #How much it swings above and below the same
    amplitude = 150
    #Sine wave: peaks during the day ,troughs at night
    #multiply by 2*pi/batches_per_day to complete one cycle per day
    daily_pattern = amplitude * np.sin(2 * np.pi * t/batches_per_day)
    #add random noise(real grids fluctuate unpredictably)
    noise = np.random.normal(0,30,num_batches)
    # combine and clip to realistic range(50-600 gcC02/kWh)
    carbon = base + daily_pattern + noise
    carbon = np.clip(carbon, 50, 600)

    return carbon
    

def generate_full_dataset():
    # This function will generate all the data by calling all the above functions and combine them into one data frame

    logger.info("Starting Full dataset generation.....")

    signals = generate_energy_signals(SIM_NUM_BATCHES,SIM_SIGNAL_LENGTH,SIM_RANDOM_SEED)
    process_df = generate_process_parameters(SIM_NUM_BATCHES,SIM_RANDOM_SEED)
    material_df = generate_material_profiles(SIM_NUM_BATCHES,SIM_RANDOM_SEED)
    targets_df = generate_targets(process_df,material_df,SIM_RANDOM_SEED)
    carbon = generate_carbon_intensity(SIM_NUM_BATCHES,SIM_RANDOM_SEED)

    master_df = pd.concat([process_df,material_df,targets_df],axis=1)

    master_df["carbon_intensity"]= carbon

    master_df["batch_id"] = [f"BATCH_{i:04d}"for i in range(SIM_NUM_BATCHES)]

    os.makedirs(SIMULATED_DIR,exist_ok=True)

    csv_path = os.path.join(SIMULATED_DIR,"batch_data.csv")
    npy_path = os.path.join(SIMULATED_DIR,"energy_signals.npy")

    master_df.to_csv(csv_path,index=False)
    np.save(npy_path,signals)

    logger.info(f"Batch data saved->{csv_path}")
    logger.info(f"Energy signals saved->{npy_path}")
    logger.info(f"DataFrame shape:  {master_df.shape}")
    logger.info(f"Signals shape:    {signals.shape}")
    logger.info(f"Yield range:      {master_df['yield'].min():.3f}->{master_df['yield'].max():.3f}")
    logger.info(f"Quality range     {master_df['quality'].min():.3f}->{master_df['quality'].max():.3f}")
    logger.info(f"Energyrange       {master_df['energy_consumption'].min():.1f}->{master_df['energy_consumption'].max():.1f}")
    logger.info(f"Carbon range      {master_df['carbon_intensity'].min():.1f}->{master_df['carbon_intensity'].max():.1f}")

    return master_df, signals



# To verify the working of simulation
# if __name__ == "__main__":
#     signals = generate_energy_signals(10, 128, 42)
#     print(f"Shape: {signals.shape}")        # Should print: Shape: (10, 128)
#     print(f"Min: {signals.min():.2f}")
#     print(f"Max: {signals.max():.2f}")



# To verify diffrent factors   in this Dataframes
# if __name__ == "__main__":
#     signals = generate_energy_signals(10, 128, 42)
#     print(f"Signals shape: {signals.shape}")      # (10, 128)

#     process_df = generate_process_parameters(10, 42)
#     print(process_df.head())                      # First 5 rows
#     print(process_df.shape)                       # (10, 5)


# Tp verify DataFrame of  material profiles
# if __name__ == "__main__":
#     signals = generate_energy_signals(10, 128, 42)
#     print(f"Signals shape: {signals.shape}")       # (10, 128)

#     process_df = generate_process_parameters(10, 42)
#     print(f"Process shape: {process_df.shape}")    # (10, 5)

#     material_df = generate_material_profiles(10, 42)
#     print(material_df.head())                      # First 5 rows
#     print(f"Material shape: {material_df.shape}")  # (10, 3)
#     print(f"Grades present: {material_df['material_grade'].unique()}")  # [1, 2, 3]


# TO check the working of the  Materials  and calculation of physical ones 
# if __name__ == "__main__":
#     signals = generate_energy_signals(10, 128, 42)
#     print(f"Signals shape: {signals.shape}")       # (10, 128)

#     process_df = generate_process_parameters(10, 42)
#     print(f"Process shape: {process_df.shape}")    # (10, 5)

#     material_df = generate_material_profiles(10, 42)
#     print(material_df.head())                      # First 5 rows
#     print(f"Material shape: {material_df.shape}")  # (10, 3)
#     print(f"Grades present: {material_df['material_grade'].unique()}")  # [1, 2, 3]


# TO verify that the simulation of carbon is working fine

# if __name__ == "__main__":
#     carbon = generate_carbon_intensity(48,42)
#     print(f"Carbon Shape: {carbon.shape}")
#     print(f"Min: {carbon.min():.1f}gCO2/kWh")
#     print(f"Max : {carbon.max():.1f}gCO2/kWh")
#     print(f"Mean : {carbon.mean():.1f}gCO2/kWh")
#     print(f"First 24 values:\n{carbon[:24].round(1)}")

# #  TO check the full data simulation
# if __name__ == "__main__":
#     df, signals = generate_full_dataset()
#     print(f"\nDataFrame shape:  {df.shape}")        # (2000, 12)
#     print(f"Signals shape:    {signals.shape}")     # (2000, 128)
#     print(f"\nColumns: {list(df.columns)}")
#     print(f"\nFirst row:\n{df.iloc[0]}")



