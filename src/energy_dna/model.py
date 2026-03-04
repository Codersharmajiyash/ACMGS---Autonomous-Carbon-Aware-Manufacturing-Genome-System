import torch 
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim,hidden_dim,latent_dim,num_layers):
        super(LSTMAutoencoder, self).__init__()
        #ENCODER
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size= hidden_dim,
            num_layers =num_layers,
            batch_first=True
        )
        self.encoder_fc= nn.Linear(hidden_dim, latent_dim)
        #DECODER
        self.decoder_fc = nn.Linear(latent_dim,hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    #you will add layers here

    def encode(self, x):
        #x shape in: (batch_size,seq_length,input_dim)
        #Run throug LSTM we only care about the hidden state
        _,(hidden,_) = self.encoder_lstm(x)
        #hidden shap: (num_layers,batch_size,hidden_dim)
        #We only want the last layers output
        hidden_last = hidden[-1]
        #compress to latent dimension
        latent = self.encoder_fc(hidden_last)#shape (batch_size,latent_dim)

        return latent
    
    def decode(self, latent,seq_length):
        #latent shape: (batch_size,latent_dim)
        batch_size = latent.size(0)

        #Expand latent back to hidden dimension: 16 -> 64
        hidden_expanded = self.decoder_fc(latent)

        # the decode LSTM needs a sequence but we only havee one vector  
        # SOlution: repeat the vector once per time stop 
        decoder_input = hidden_expanded.unsqueeze(1).repeat(1, seq_length, 1)
        # unsqueeze->(batch_size,seq length, hidden_dim)
        # repeat(1, seq_length)
        # Run through decoder LSTM
        decoder_output,_ = self.decoder_lstm(decoder_input)
        # shape : (batch_size, seq_elength, hidden)dim)
        # Project back to original signal dimenesion: 64 -1
        reconstructed = self.output_layer(decoder_output)
        # Shaoe: (bathc_size,seq_length,input_dim)
        return reconstructed
    def forward(self, x):
        # x shape: (batch_size,seq_length , input_dim)
        seq_length = x.size(1)

        latent = self.encode(x)
        reconstructed = self.decode(latent, seq_length)

        return reconstructed, latent



if __name__ == "__main__":
    from config.settings import (
        ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM,
        ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS
    )

    model = LSTMAutoencoder(
        input_dim=ENERGY_INPUT_DIM,
        hidden_dim=ENERGY_HIDDEN_DIM,
        latent_dim=ENERGY_LATENT_DIM,
        num_layers=ENERGY_NUM_LAYERS
    )

    print(model)   # Print model architecture

    # Dummy batch: 8 signals, 128 time steps, 1 feature
    dummy = torch.randn(8, 128, 1)
    reconstructed, latent = model(dummy)

    print(f"\nInput shape:         {dummy.shape}")         # (8, 128, 1)
    print(f"Latent shape:        {latent.shape}")          # (8, 16)
    print(f"Reconstructed shape: {reconstructed.shape}")   # (8, 128, 1)