"""Atom-level SMILES tokenizer with vocabulary management."""

import json
import re
from pathlib import Path
from typing import List, Optional


class SMILESTokenizer:
    """Atom-level tokenizer for SMILES strings.

    Uses regex to extract chemically meaningful tokens like Cl, Br, [nH], etc.

    Special tokens:
        <PAD>: Padding token (index 0)
        <SOS>: Start of sequence (index 1)
        <EOS>: End of sequence (index 2)
        <UNK>: Unknown token (index 3)
    """

    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    # Regex pattern for atom-level tokenization
    # Matches: bracketed atoms, Br, Cl, two-char atoms, single chars, etc.
    PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}
        self.vocab_size: int = 0
        self._regex = re.compile(self.PATTERN)

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into list of atom-level tokens."""
        return self._regex.findall(smiles)

    def build_vocab(self, smiles_list: List[str]) -> None:
        """Build vocabulary from list of SMILES strings."""
        tokens = set()
        for smiles in smiles_list:
            tokens.update(self.tokenize(smiles))

        # Special tokens first, then sorted vocabulary
        all_tokens = self.SPECIAL_TOKENS + sorted(tokens)
        self.token2idx = {t: i for i, t in enumerate(all_tokens)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}
        self.vocab_size = len(all_tokens)

    def encode(self, smiles: str, add_special: bool = True) -> List[int]:
        """Encode SMILES string to list of token indices.

        Args:
            smiles: SMILES string to encode
            add_special: Whether to add <SOS> and <EOS> tokens

        Returns:
            List of token indices
        """
        tokens = self.tokenize(smiles)
        indices = [self.token2idx.get(t, self.token2idx["<UNK>"]) for t in tokens]

        if add_special:
            indices = [self.token2idx["<SOS>"]] + indices + [self.token2idx["<EOS>"]]

        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decode list of indices back to SMILES string.

        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens

        Returns:
            Decoded SMILES string
        """
        tokens = [self.idx2token.get(i, "") for i in indices]

        if remove_special:
            tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]

        return "".join(tokens)

    @property
    def pad_idx(self) -> int:
        return self.token2idx["<PAD>"]

    @property
    def sos_idx(self) -> int:
        return self.token2idx["<SOS>"]

    @property
    def eos_idx(self) -> int:
        return self.token2idx["<EOS>"]

    def save(self, path: Path) -> None:
        """Save vocabulary to JSON file."""
        path = Path(path)
        data = {
            "token2idx": self.token2idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load vocabulary from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        self.token2idx = data["token2idx"]
        self.idx2token = {int(i): t for t, i in self.token2idx.items()}
        self.vocab_size = data["vocab_size"]
