"""PairFormer model for flow matching molecular generation.

Wraps PairFormer components in a model with the same interface as GraphDiT,
allowing drop-in replacement for architecture comparison experiments.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pairformer import PairFormerBlock
from src.models.dit import SinusoidalPositionalEmbedding


class PairFormerFlow(nn.Module):
    """PairFormer model for molecular graph generation via flow matching.

    Maintains separate node (s) and edge (z) tracks with explicit interactions.
    This architecture is inspired by AlphaFold2 and adapted for molecular graphs.

    Architecture:
        Input: (x, adj, t) where x is node features, adj is adjacency, t is timestep
        1. Embed nodes → s track
        2. Embed edges → z track
        3. Process through PairFormer blocks with timestep conditioning
        4. Project back to node/edge velocities

    Args:
        num_atom_types: Number of atom types
        num_bond_types: Number of bond types (including no-bond)
        max_atoms: Maximum atoms per molecule
        hidden_dim_s: Node track hidden dimension
        hidden_dim_z: Edge track hidden dimension
        num_layers: Number of PairFormer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        t_embed_dim: Timestep embedding dimension
        use_triangle_mult: Whether to use triangle multiplication
        use_triangle_attn: Whether to use triangle attention
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        max_atoms: int,
        hidden_dim_s: int = 256,
        hidden_dim_z: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        t_embed_dim: Optional[int] = None,
        use_triangle_mult: bool = True,
        use_triangle_attn: bool = True,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.max_atoms = max_atoms
        self.hidden_dim_s = hidden_dim_s
        self.hidden_dim_z = hidden_dim_z
        self.num_layers = num_layers

        if t_embed_dim is None:
            t_embed_dim = hidden_dim_s

        # Input projections
        self.node_embed = nn.Linear(num_atom_types, hidden_dim_s)
        self.edge_embed = nn.Linear(num_bond_types, hidden_dim_z)

        # Positional embeddings (learnable, separate for nodes and edges)
        self.node_pos_embed = nn.Parameter(
            torch.randn(1, max_atoms, hidden_dim_s) * 0.02
        )
        # Edge positional embeddings based on (i, j) indices
        self.edge_pos_embed_row = nn.Parameter(
            torch.randn(1, max_atoms, 1, hidden_dim_z // 2) * 0.02
        )
        self.edge_pos_embed_col = nn.Parameter(
            torch.randn(1, 1, max_atoms, hidden_dim_z // 2) * 0.02
        )

        # Timestep embedding network
        self.t_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, hidden_dim_s * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim_s * 4, hidden_dim_s),
        )

        # PairFormer blocks
        self.blocks = nn.ModuleList([
            PairFormerBlock(
                d_s=hidden_dim_s,
                d_z=hidden_dim_z,
                cond_dim=hidden_dim_s,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_triangle_mult=use_triangle_mult,
                use_triangle_attn=use_triangle_attn,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm_s = nn.LayerNorm(hidden_dim_s)
        self.final_norm_z = nn.LayerNorm(hidden_dim_z)

        # Output projections
        self.out_node = nn.Linear(hidden_dim_s, num_atom_types)
        self.out_edge = nn.Linear(hidden_dim_z, num_bond_types)

        # Initialize output layers to zero for better training stability
        nn.init.zeros_(self.out_node.weight)
        nn.init.zeros_(self.out_node.bias)
        nn.init.zeros_(self.out_edge.weight)
        nn.init.zeros_(self.out_edge.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity field for flow matching.

        Args:
            x: Node features [batch, max_atoms, num_atom_types]
            adj: Adjacency features [batch, max_atoms, max_atoms, num_bond_types]
            t: Timesteps [batch] in [0, 1]
            mask: Node mask [batch, max_atoms] - True for valid atoms

        Returns:
            v_x: Node velocity [batch, max_atoms, num_atom_types]
            v_adj: Edge velocity [batch, max_atoms, max_atoms, num_bond_types]
        """
        B, N, A = x.shape
        _, N1, N2, E = adj.shape

        assert N == self.max_atoms, f"Expected {self.max_atoms} atoms, got {N}"
        assert N1 == N2 == N, f"Adjacency shape mismatch"

        # Embed timestep
        t_emb = self.t_embed(t)  # [B, hidden_dim_s]

        # Project inputs to hidden dimensions
        s = self.node_embed(x)  # [B, N, hidden_dim_s]
        z = self.edge_embed(adj)  # [B, N, N, hidden_dim_z]

        # Add positional embeddings
        s = s + self.node_pos_embed

        # Edge positional embedding: combine row and column embeddings
        edge_pos = torch.cat([
            self.edge_pos_embed_row.expand(B, -1, N, -1),
            self.edge_pos_embed_col.expand(B, N, -1, -1),
        ], dim=-1)
        z = z + edge_pos

        # Apply PairFormer blocks
        for block in self.blocks:
            s, z = block(s, z, t_emb, mask)

        # Final normalization
        s = self.final_norm_s(s)
        z = self.final_norm_z(z)

        # Project to output dimensions
        v_x = self.out_node(s)  # [B, N, num_atom_types]
        v_adj = self.out_edge(z)  # [B, N, N, num_bond_types]

        return v_x, v_adj

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_pairformer_from_config(config: dict) -> PairFormerFlow:
    """Create PairFormerFlow model from configuration dictionary.

    Args:
        config: Configuration dictionary with 'model' and 'graph' sections

    Returns:
        Initialized PairFormerFlow model
    """
    graph_config = config.get("graph", {})
    model_config = config.get("model", {})

    return PairFormerFlow(
        num_atom_types=len(graph_config.get("atom_types", ["C", "N", "O", "F", "H"])),
        num_bond_types=len(
            graph_config.get("bond_types", ["none", "single", "double", "triple", "aromatic"])
        ),
        max_atoms=graph_config.get("max_atoms", 9),
        hidden_dim_s=model_config.get("hidden_dim_s", model_config.get("hidden_dim", 256)),
        hidden_dim_z=model_config.get("hidden_dim_z", 128),
        num_layers=model_config.get("num_layers", 8),
        num_heads=model_config.get("num_heads", 8),
        mlp_ratio=model_config.get("mlp_ratio", 4.0),
        dropout=model_config.get("dropout", 0.1),
        t_embed_dim=model_config.get("t_embed_dim"),
        use_triangle_mult=model_config.get("use_triangle_mult", True),
        use_triangle_attn=model_config.get("use_triangle_attn", True),
    )


def test_pairformer_flow():
    """Test PairFormerFlow model."""
    print("Testing PairFormerFlow...")

    batch_size = 4
    max_atoms = 9
    num_atom_types = 5
    num_bond_types = 5
    hidden_dim_s = 128
    hidden_dim_z = 64
    num_layers = 2
    num_heads = 4

    x = torch.randn(batch_size, max_atoms, num_atom_types)
    adj = torch.randn(batch_size, max_atoms, max_atoms, num_bond_types)
    t = torch.rand(batch_size)
    mask = torch.ones(batch_size, max_atoms).bool()
    mask[:, 7:] = False

    # Test full PairFormer
    print("\n1. Full PairFormer:")
    model = PairFormerFlow(
        num_atom_types=num_atom_types,
        num_bond_types=num_bond_types,
        max_atoms=max_atoms,
        hidden_dim_s=hidden_dim_s,
        hidden_dim_z=hidden_dim_z,
        num_layers=num_layers,
        num_heads=num_heads,
        use_triangle_mult=True,
        use_triangle_attn=True,
    )
    v_x, v_adj = model(x, adj, t, mask)
    print(f"   v_x shape: {v_x.shape}")
    print(f"   v_adj shape: {v_adj.shape}")
    print(f"   Parameters: {model.count_parameters():,}")
    assert v_x.shape == (batch_size, max_atoms, num_atom_types)
    assert v_adj.shape == (batch_size, max_atoms, max_atoms, num_bond_types)
    print("   PASSED")

    # Test minimal PairFormer (pair bias only)
    print("\n2. Minimal PairFormer (pair bias only):")
    model_min = PairFormerFlow(
        num_atom_types=num_atom_types,
        num_bond_types=num_bond_types,
        max_atoms=max_atoms,
        hidden_dim_s=hidden_dim_s,
        hidden_dim_z=hidden_dim_z,
        num_layers=num_layers,
        num_heads=num_heads,
        use_triangle_mult=False,
        use_triangle_attn=False,
    )
    v_x, v_adj = model_min(x, adj, t, mask)
    print(f"   v_x shape: {v_x.shape}")
    print(f"   v_adj shape: {v_adj.shape}")
    print(f"   Parameters: {model_min.count_parameters():,}")
    print("   PASSED")

    # Test with triangle mult only
    print("\n3. PairFormer with triangle mult only:")
    model_mult = PairFormerFlow(
        num_atom_types=num_atom_types,
        num_bond_types=num_bond_types,
        max_atoms=max_atoms,
        hidden_dim_s=hidden_dim_s,
        hidden_dim_z=hidden_dim_z,
        num_layers=num_layers,
        num_heads=num_heads,
        use_triangle_mult=True,
        use_triangle_attn=False,
    )
    v_x, v_adj = model_mult(x, adj, t, mask)
    print(f"   v_x shape: {v_x.shape}")
    print(f"   v_adj shape: {v_adj.shape}")
    print(f"   Parameters: {model_mult.count_parameters():,}")
    print("   PASSED")

    # Test config-based creation
    print("\n4. Create from config:")
    config = {
        "graph": {
            "atom_types": ["C", "N", "O", "F", "H"],
            "bond_types": ["none", "single", "double", "triple", "aromatic"],
            "max_atoms": 9,
        },
        "model": {
            "hidden_dim_s": 128,
            "hidden_dim_z": 64,
            "num_layers": 2,
            "num_heads": 4,
            "use_triangle_mult": True,
            "use_triangle_attn": True,
        },
    }
    model_cfg = create_pairformer_from_config(config)
    v_x, v_adj = model_cfg(x, adj, t, mask)
    print(f"   Created from config successfully")
    print(f"   Parameters: {model_cfg.count_parameters():,}")
    print("   PASSED")

    print("\nAll PairFormerFlow tests passed!")


if __name__ == "__main__":
    test_pairformer_flow()
