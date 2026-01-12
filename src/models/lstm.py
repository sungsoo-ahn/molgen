"""LSTM-based language model for SMILES generation."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tokenizer import SMILESTokenizer


class SMILESLSTM(nn.Module):
    """LSTM language model for SMILES generation.

    Architecture:
        Embedding -> LSTM (multi-layer) -> Dropout -> Linear

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        pad_idx: Padding token index
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # LSTM layers
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input token indices [batch, seq_len]
            hidden: Optional initial hidden state (h, c)

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            hidden: Final hidden state (h, c)
        """
        # Embed input
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # LSTM forward
        output, hidden = self.lstm(embedded, hidden)  # [batch, seq_len, hidden_dim]

        # Project to vocabulary
        output = self.dropout(output)
        logits = self.fc(output)  # [batch, seq_len, vocab_size]

        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    @torch.no_grad()
    def generate(
        self,
        tokenizer: SMILESTokenizer,
        num_samples: int = 1,
        max_len: int = 100,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> List[str]:
        """Generate SMILES strings using temperature sampling.

        Args:
            tokenizer: SMILESTokenizer instance
            num_samples: Number of SMILES to generate
            max_len: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            device: Device to run on

        Returns:
            List of generated SMILES strings
        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        generated = []

        for _ in range(num_samples):
            hidden = self.init_hidden(1, device)
            current_token = torch.tensor([[tokenizer.sos_idx]], device=device)
            tokens = []

            for _ in range(max_len):
                logits, hidden = self.forward(current_token, hidden)
                logits = logits[:, -1, :] / temperature

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_idx = next_token.item()

                # Stop at EOS
                if token_idx == tokenizer.eos_idx:
                    break

                # Skip PAD tokens
                if token_idx != tokenizer.pad_idx:
                    tokens.append(token_idx)

                current_token = next_token

            smiles = tokenizer.decode(tokens, remove_special=True)
            generated.append(smiles)

        return generated
