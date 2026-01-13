"""Molecular graph conversion utilities for flow matching models.

Converts molecules between RDKit Mol objects and continuous tensor representations
for use with flow matching generative models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import BondType
from torch.utils.data import Dataset


# Default atom types for different datasets
ATOM_TYPES_QM9 = ["C", "N", "O", "F", "H"]
ATOM_TYPES_ZINC = ["C", "N", "O", "F", "S", "Cl", "Br", "P", "I"]

# Bond types (None represents no bond)
BOND_TYPES = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


class MolecularGraphConverter:
    """Convert molecules between RDKit Mol objects and continuous tensors.

    This converter handles bidirectional transformation between molecular
    representations for flow matching models operating on 2D molecular graphs.

    Args:
        atom_types: List of atom symbols to support (e.g., ['C', 'N', 'O', 'F', 'H'])
        bond_types: List of bond types (None for no bond, then RDKit BondType values)
        max_atoms: Maximum number of atoms per molecule (for padding)

    Attributes:
        atom_to_idx: Dict mapping atom symbol to index
        idx_to_atom: Dict mapping index to atom symbol
        bond_to_idx: Dict mapping bond type to index
        idx_to_bond: Dict mapping index to bond type
    """

    def __init__(
        self,
        atom_types: List[str],
        bond_types: List[Optional[BondType]] = BOND_TYPES,
        max_atoms: int = 9,
    ):
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_atoms = max_atoms

        # Build mappings
        self.atom_to_idx: Dict[str, int] = {atom: i for i, atom in enumerate(atom_types)}
        self.idx_to_atom: Dict[int, str] = {i: atom for i, atom in enumerate(atom_types)}

        self.bond_to_idx: Dict[Optional[BondType], int] = {
            bond: i for i, bond in enumerate(bond_types)
        }
        self.idx_to_bond: Dict[int, Optional[BondType]] = {
            i: bond for i, bond in enumerate(bond_types)
        }

    @property
    def num_atom_types(self) -> int:
        """Number of supported atom types."""
        return len(self.atom_types)

    @property
    def num_bond_types(self) -> int:
        """Number of supported bond types."""
        return len(self.bond_types)

    def mol_to_tensors(
        self, mol: Chem.Mol
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert RDKit Mol to tensor representation.

        Args:
            mol: RDKit Mol object

        Returns:
            x: Node features [max_atoms, num_atom_types] - one-hot encoding
            adj: Adjacency matrix [max_atoms, max_atoms, num_bond_types] - one-hot edges
            mask: Node mask [max_atoms] - True for real atoms, False for padding

        Raises:
            ValueError: If molecule has more atoms than max_atoms or unknown atom type
        """
        num_atoms = mol.GetNumAtoms()

        if num_atoms > self.max_atoms:
            raise ValueError(
                f"Molecule has {num_atoms} atoms, max is {self.max_atoms}"
            )

        # Initialize tensors
        x = torch.zeros(self.max_atoms, self.num_atom_types, dtype=torch.float32)
        adj = torch.zeros(
            self.max_atoms, self.max_atoms, self.num_bond_types, dtype=torch.float32
        )
        mask = torch.zeros(self.max_atoms, dtype=torch.bool)

        # Fill node features (one-hot atom types)
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if symbol not in self.atom_to_idx:
                raise ValueError(f"Unknown atom type: {symbol}")
            x[i, self.atom_to_idx[symbol]] = 1.0
            mask[i] = True

        # Fill edge features (one-hot bond types)
        # Initialize with "no bond" for all pairs
        adj[:, :, 0] = 1.0

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            if bond_type not in self.bond_to_idx:
                # Default to single bond for unknown types
                bond_idx = self.bond_to_idx[BondType.SINGLE]
            else:
                bond_idx = self.bond_to_idx[bond_type]

            # Symmetric adjacency
            adj[i, j, 0] = 0.0  # Remove "no bond"
            adj[j, i, 0] = 0.0
            adj[i, j, bond_idx] = 1.0
            adj[j, i, bond_idx] = 1.0

        return x, adj, mask

    def tensors_to_mol(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[Chem.Mol]:
        """Convert tensor representation back to RDKit Mol.

        Applies argmax to discretize continuous predictions, then constructs
        a molecule. Returns None if the molecule is invalid.

        Args:
            x: Node features [max_atoms, num_atom_types] or [N, num_atom_types]
            adj: Adjacency matrix [max_atoms, max_atoms, num_bond_types]
            mask: Node mask [max_atoms] - True for real atoms

        Returns:
            RDKit Mol object, or None if molecule is invalid
        """
        # Discretize node features
        atom_indices = x.argmax(dim=-1)  # [max_atoms]

        # Count valid atoms
        num_atoms = mask.sum().item()
        if num_atoms == 0:
            return None

        # Create editable molecule
        mol = Chem.RWMol()

        # Add atoms
        atom_map = {}  # Map from tensor index to mol atom index
        for i in range(num_atoms):
            if mask[i]:
                atom_idx = atom_indices[i].item()
                atom_symbol = self.idx_to_atom.get(atom_idx, "C")
                mol_idx = mol.AddAtom(Chem.Atom(atom_symbol))
                atom_map[i] = mol_idx

        # Add bonds (only upper triangle to avoid duplicates)
        for i in range(num_atoms):
            if not mask[i]:
                continue
            for j in range(i + 1, num_atoms):
                if not mask[j]:
                    continue

                # Discretize bond type
                bond_idx = adj[i, j].argmax().item()

                if bond_idx == 0:
                    # No bond
                    continue

                bond_type = self.idx_to_bond.get(bond_idx, BondType.SINGLE)
                if bond_type is not None:
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)

        # Try to sanitize the molecule
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def smiles_to_tensors(
        self, smiles: str, kekulize: bool = False
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Convert SMILES string to tensor representation.

        Args:
            smiles: SMILES string
            kekulize: If True, convert aromatic bonds to single/double bonds

        Returns:
            Tuple of (x, adj, mask) tensors, or None if SMILES is invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            if kekulize:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            return self.mol_to_tensors(mol)
        except ValueError:
            return None
        except Exception:
            # Kekulization can fail for some molecules
            return None

    def tensors_to_smiles(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[str]:
        """Convert tensor representation to SMILES string.

        Args:
            x: Node features [max_atoms, num_atom_types]
            adj: Adjacency matrix [max_atoms, max_atoms, num_bond_types]
            mask: Node mask [max_atoms]

        Returns:
            SMILES string, or None if molecule is invalid
        """
        mol = self.tensors_to_mol(x, adj, mask)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    def save(self, path: Path) -> None:
        """Save converter configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        config = {
            "atom_types": self.atom_types,
            "bond_types": [str(b) if b is not None else None for b in self.bond_types],
            "max_atoms": self.max_atoms,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MolecularGraphConverter":
        """Load converter configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            MolecularGraphConverter instance
        """
        with open(path, "r") as f:
            config = json.load(f)

        # Convert bond type strings back to BondType enums
        bond_types = []
        for b in config["bond_types"]:
            if b is None:
                bond_types.append(None)
            elif b == "SINGLE":
                bond_types.append(BondType.SINGLE)
            elif b == "DOUBLE":
                bond_types.append(BondType.DOUBLE)
            elif b == "TRIPLE":
                bond_types.append(BondType.TRIPLE)
            elif b == "AROMATIC":
                bond_types.append(BondType.AROMATIC)
            else:
                bond_types.append(None)

        return cls(
            atom_types=config["atom_types"],
            bond_types=bond_types,
            max_atoms=config["max_atoms"],
        )


class MolecularGraphDataset(Dataset):
    """PyTorch Dataset for molecular graphs.

    Args:
        smiles_list: List of SMILES strings
        converter: MolecularGraphConverter instance
        skip_invalid: If True, skip invalid molecules silently
    """

    def __init__(
        self,
        smiles_list: List[str],
        converter: MolecularGraphConverter,
        skip_invalid: bool = True,
    ):
        self.converter = converter
        self.valid_indices: List[int] = []
        self.smiles_list = smiles_list

        # Pre-validate and cache valid indices
        for i, smiles in enumerate(smiles_list):
            result = converter.smiles_to_tensors(smiles)
            if result is not None:
                self.valid_indices.append(i)
            elif not skip_invalid:
                raise ValueError(f"Invalid SMILES at index {i}: {smiles}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles_idx = self.valid_indices[idx]
        smiles = self.smiles_list[smiles_idx]

        x, adj, mask = self.converter.smiles_to_tensors(smiles)

        return {
            "node_features": x,  # [max_atoms, num_atom_types]
            "adjacency": adj,  # [max_atoms, max_atoms, num_bond_types]
            "mask": mask,  # [max_atoms]
        }


def get_size_distribution(
    smiles_list: List[str],
    converter: MolecularGraphConverter,
) -> Dict[int, float]:
    """Compute molecule size distribution from SMILES list.

    Args:
        smiles_list: List of SMILES strings
        converter: MolecularGraphConverter instance

    Returns:
        Dictionary mapping number of atoms to probability
    """
    size_counts: Dict[int, int] = {}

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            if num_atoms <= converter.max_atoms:
                size_counts[num_atoms] = size_counts.get(num_atoms, 0) + 1

    total = sum(size_counts.values())
    return {size: count / total for size, count in size_counts.items()}


def sample_molecule_sizes(
    size_distribution: Dict[int, float],
    num_samples: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample molecule sizes from distribution.

    Args:
        size_distribution: Dictionary mapping size to probability
        num_samples: Number of samples to generate
        device: Device for output tensor

    Returns:
        Tensor of molecule sizes [num_samples]
    """
    sizes = list(size_distribution.keys())
    probs = [size_distribution[s] for s in sizes]

    # Sample indices
    indices = torch.multinomial(
        torch.tensor(probs, device=device),
        num_samples=num_samples,
        replacement=True,
    )

    # Map to sizes
    sizes_tensor = torch.tensor(sizes, device=device)
    return sizes_tensor[indices]


class PreprocessedMolecularGraphDataset(Dataset):
    """PyTorch Dataset for preprocessed molecular graph tensors.

    Loads pre-converted tensors from a .pt file for fast iteration.

    Args:
        tensor_path: Path to .pt file with preprocessed tensors
    """

    def __init__(self, tensor_path: Path):
        data = torch.load(tensor_path, weights_only=True)
        self.node_features = data["node_features"]
        self.adjacency = data["adjacency"]
        self.mask = data["mask"]

    def __len__(self) -> int:
        return len(self.node_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "node_features": self.node_features[idx],
            "adjacency": self.adjacency[idx],
            "mask": self.mask[idx],
        }


def create_mask_from_sizes(
    sizes: torch.Tensor,
    max_atoms: int,
) -> torch.Tensor:
    """Create mask tensor from molecule sizes.

    Args:
        sizes: Tensor of molecule sizes [batch_size]
        max_atoms: Maximum atoms for padding

    Returns:
        Mask tensor [batch_size, max_atoms] with True for valid atoms
    """
    batch_size = sizes.shape[0]
    device = sizes.device

    # Create range tensor
    indices = torch.arange(max_atoms, device=device).unsqueeze(0)  # [1, max_atoms]
    sizes_expanded = sizes.unsqueeze(1)  # [batch_size, 1]

    # Mask where index < size
    return indices < sizes_expanded
