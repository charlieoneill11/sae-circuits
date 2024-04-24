import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
import math
from fancy_einsum import einsum
from vector_quantize_pytorch import VectorQuantize, FSQ


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Quantizer
        levels = [8, 8, 8, 5, 5, 5] # see 4.1 and A.4.1 in the paper
        self.quantizer = FSQ(levels)
        self.bottleneck_dim = len(levels)

        # Positional Encoding
        self.positional_encoder = PositionalEncoding(d_model, dropout)

        # Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Linear projection down - replaces torch.zeros with nn.Linear
        self.encoder_output_projection = nn.Linear(d_model, self.bottleneck_dim)

        # Linear projection up - replaces torch.zeros with nn.Linear
        self.decoder_input_projection = nn.Linear(self.bottleneck_dim, d_model)

        # Decoder Layer
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        self.encoder_input_projection = nn.Linear(input_dim, d_model)
        self.decoder_output_projection = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # Encode
        src = self.encoder_input_projection(src)
        src = self.positional_encoder(src)
        memory = self.transformer_encoder(src)

        # Apply the encoder output projection down
        memory = F.relu(self.encoder_output_projection(memory))

        # Vector quantize the memory
        quantized_memory, _, commit_loss = self.quantize(memory)

        # Decode
        quantized_memory = F.relu(self.decoder_input_projection(quantized_memory))
        output = self.transformer_decoder(quantized_memory, quantized_memory)
        output = self.decoder_output_projection(output)
        return output, commit_loss
    
    def encode(self, src):

        # Encode
        src = self.encoder_input_projection(src)
        src = self.positional_encoder(src)
        memory = self.transformer_encoder(src)

        # Apply the encoder output projection down
        memory = F.relu(self.encoder_output_projection(memory))

        return memory

    def quantize(self, bottleneck):
        quantized, indices = self.quantizer(bottleneck)
        commit_loss = torch.tensor(0).unsqueeze(0)
        return quantized, indices, commit_loss

    def quantized_indices(self, src):
        # Encode
        src = self.encoder_input_projection(src)
        src = self.positional_encoder(src)
        memory = self.transformer_encoder(src)

        # Apply the encoder output projection down
        memory = F.relu(self.encoder_output_projection(memory))

        # Vector quantize the memory
        quantised, indices, _ = self.quantize(memory)

        return quantised, indices

    def indices_to_rep(self, indices):
        with torch.no_grad():
            low_dim_vectors = self.quantizer.get_codes_from_indices([indices])
            quantized_memory = self.quantizer.project_out(low_dim_vectors)
            # Decode
            quantized_memory = F.relu(self.decoder_input_projection(quantized_memory))
            output = self.transformer_decoder(quantized_memory, quantized_memory)
            output = self.decoder_output_projection(output)
            return output


class TransformerVQVAE(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 codebook_size=128, codebook_dim=16, threshold_ema_dead_code=2, dropout=0.1):
        super(TransformerVQVAE, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # VQ Quantizer
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.quantizer = VectorQuantize(
            dim=self.d_model,  # Assuming the dimensionality to match d_model for simplicity
            codebook_size=codebook_size,  # Example codebook size
            codebook_dim=codebook_dim,  # This is an illustrative example, adjust based on your model's needs
            decay=0.8,
            commitment_weight=1.0,
            use_cosine_sim=True,  # Example, adjust as needed
            threshold_ema_dead_code = threshold_ema_dead_code
        )
        self.bottleneck_dim = self.d_model

        # Positional Encoding
        self.positional_encoder = PositionalEncoding(d_model, dropout)

        # Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Linear projection down - replaces torch.zeros with nn.Linear
        self.encoder_output_projection = nn.Linear(d_model, self.bottleneck_dim)

        # Linear projection up - replaces torch.zeros with nn.Linear
        self.decoder_input_projection = nn.Linear(self.bottleneck_dim, d_model)

        # Decoder Layer
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        self.encoder_input_projection = nn.Linear(input_dim, d_model)
        self.decoder_output_projection = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # Encode
        src = self.encoder_input_projection(src)
        src = self.positional_encoder(src)
        memory = self.transformer_encoder(src)
        
        # Apply the encoder output projection down
        quantized_memory = F.relu(self.encoder_output_projection(memory))

        # Vector quantize the memory
        quantized_memory, _, commit_loss = quantized_memory, 0, torch.tensor(0.0).unsqueeze(0)
        #quantized_memory, _, commit_loss = self.quantize(quantized_memory)

        # Decode
        quantized_memory = F.relu(self.decoder_input_projection(quantized_memory))
        output = self.transformer_decoder(quantized_memory, quantized_memory)
        output = self.decoder_output_projection(output)
        return output, commit_loss

    def encode(self, src):

        # Encode
        src = self.encoder_input_projection(src)
        src = self.positional_encoder(src)
        memory = self.transformer_encoder(src)

        # Apply the encoder output projection down
        memory = F.relu(self.encoder_output_projection(memory))

        return memory

    def quantize(self, bottleneck):
        quantized, indices, commit_loss = self.quantizer(bottleneck)
        return quantized, indices, commit_loss

    def quantized_indices(self, src):
        # Encode
        src = self.encoder_input_projection(src)
        src = self.positional_encoder(src)
        memory = self.transformer_encoder(src)
        
        # Apply the encoder output projection down
        memory = F.relu(self.encoder_output_projection(memory))

        # Vector quantize the memory
        quantised, indices, _ = self.quantize(memory)

        return quantised, indices
    
    def indices_to_rep(self, indices):
        with torch.no_grad():
            low_dim_vectors = self.quantizer.get_codes_from_indices([indices])
            quantized_memory = self.quantizer.project_out(low_dim_vectors)
            # Decode
            quantized_memory = F.relu(self.decoder_input_projection(quantized_memory))
            output = self.transformer_decoder(quantized_memory, quantized_memory)
            output = self.decoder_output_projection(output)
            return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)