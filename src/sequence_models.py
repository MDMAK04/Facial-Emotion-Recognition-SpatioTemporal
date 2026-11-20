import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TemporalSequenceClassifier(nn.Module):
    def __init__(
        self,
        model_type: str = "lstm",
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.3,
        nhead: int = 4
    ):
        super().__init__()
        self.model_type = model_type.lower()

        if self.model_type == "rnn":
            self.seq_model = nn.RNN(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
            out_dim = hidden_dim

        elif self.model_type == "lstm":
            self.seq_model = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            out_dim = hidden_dim * 2

        elif self.model_type == "gru":
            self.seq_model = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            out_dim = hidden_dim * 2

        elif self.model_type == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            self.seq_model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            out_dim = hidden_dim
        else:
            raise ValueError("model_type doit être : 'rnn', 'lstm', 'gru', ou 'transformer'")

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type in ["rnn", "lstm", "gru"]:
            output, _ = self.seq_model(x)
            pooled = output.mean(dim=1)
        elif self.model_type == "transformer":
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            output = self.seq_model(x)
            pooled = output.mean(dim=1)
        return self.head(pooled)