"""PairMixer architecture for molecular graph generation.

Implements PairMixer from "Triangle Multiplication is All You Need for
Biomolecular Structure Representations" (arXiv:2510.18870).

Key differences from PairFormer:
- Edge-only backbone processing (no per-layer node updates)
- No triangle attention (only triangle multiplication)
- Node features derived from edges via attention aggregation at the end

This results in ~4x faster inference and 34% lower training cost.

Reference: https://github.com/genesistherapeutics/pairmixer
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pairformer import AdaLNPair, TriangleMultiplication
from src.models.dit import SinusoidalPositionalEmbedding


class NodeFromEdgeAttention(nn.Module):
    """Aggregate edge features to node features via attention.

    For each node i, uses attention to aggregate information from edges (i, j)
    for all j. Query comes from node features, keys/values from edge features.

    Args:
        d_s: Node feature dimension
        d_z: Edge feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_s: int,
        d_z: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads

        assert d_s % num_heads == 0, f"d_s ({d_s}) must be divisible by num_heads ({num_heads})"

        # Query from node features
        self.q_proj = nn.Linear(d_s, d_s)

        # Key/Value from edge features
        self.k_proj = nn.Linear(d_z, d_s)
        self.v_proj = nn.Linear(d_z, d_s)

        # Output projection
        self.out_proj = nn.Linear(d_s, d_s)

        self.dropout = nn.Dropout(dropout)
        self.norm_z = nn.LayerNorm(d_z)
        self.norm_s = nn.LayerNorm(d_s)

    def forward(
        self,
        z: torch.Tensor,
        s_query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate edge features to node features.

        Args:
            z: Edge features [batch, N, N, d_z]
            s_query: Node features for queries [batch, N, d_s]
            mask: Node validity mask [batch, N] - True for valid

        Returns:
            Node features [batch, N, d_s]
        """
        B, N, _, _ = z.shape

        # Normalize inputs
        z = self.norm_z(z)
        s_query = self.norm_s(s_query)

        # For each node i, we want to attend over edges z[i, j] for all j
        # Reshape z to [B*N, N, d_z] where each "batch" is edges from node i
        z_flat = z.reshape(B * N, N, self.d_z)

        # Query: [B, N, d_s] -> [B*N, 1, d_s] (each node queries once)
        q = self.q_proj(s_query).view(B * N, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Key/Value from edges: [B*N, N, d_z] -> [B*N, N, d_s]
        k = self.k_proj(z_flat).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z_flat).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention: [B*N, num_heads, 1, N]
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for each source node: [B, N] -> [B*N, N]
            attn_mask = mask.unsqueeze(1).expand(-1, N, -1).reshape(B * N, N)
            attn_mask = ~attn_mask.unsqueeze(1).unsqueeze(2)  # [B*N, 1, 1, N]
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention: [B*N, num_heads, 1, head_dim]
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B * N, 1, self.d_s)

        # Reshape back: [B*N, 1, d_s] -> [B, N, d_s]
        out = out.view(B, N, self.d_s)

        return self.out_proj(out)


class PairMixerBlock(nn.Module):
    """Single PairMixer block with edge-only processing.

    Processing order (edge-only):
    1. z <- z + TriMulOutgoing(AdaLN(z, t))
    2. z <- z + TriMulIncoming(AdaLN(z, t))
    3. z <- z + FFN(AdaLN(z, t))

    No node updates during backbone processing.

    Args:
        d_z: Edge feature dimension
        cond_dim: Conditioning dimension (for timestep)
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_z: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_z = d_z

        # Triangle multiplication layers
        self.norm_z1 = AdaLNPair(d_z, cond_dim)
        self.triangle_mult_out = TriangleMultiplication(d_z, outgoing=True, dropout=dropout)

        self.norm_z2 = AdaLNPair(d_z, cond_dim)
        self.triangle_mult_in = TriangleMultiplication(d_z, outgoing=False, dropout=dropout)

        # FFN
        self.norm_z3 = AdaLNPair(d_z, cond_dim)
        mlp_hidden = int(d_z * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_z, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_z),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass (edge-only).

        Args:
            z: Edge features [batch, N, N, d_z]
            t_emb: Timestep embedding [batch, cond_dim]
            mask: Node validity mask [batch, N]

        Returns:
            Updated edge features [batch, N, N, d_z]
        """
        # Triangle multiplication (outgoing)
        z = z + self.triangle_mult_out(self.norm_z1(z, t_emb), mask)

        # Triangle multiplication (incoming)
        z = z + self.triangle_mult_in(self.norm_z2(z, t_emb), mask)

        # FFN
        z = z + self.mlp(self.norm_z3(z, t_emb))

        return z


class PairMixerFlow(nn.Module):
    """PairMixer model for molecular graph generation via flow matching.

    Edge-only backbone with node features derived at the end via attention.

    Architecture:
        1. Project inputs to hidden dimensions
        2. Add positional embeddings
        3. Process through PairMixer blocks (edge-only)
        4. Aggregate edge features to nodes via attention
        5. Project to output velocities

    Args:
        num_atom_types: Number of atom types
        num_bond_types: Number of bond types (including no-bond)
        max_atoms: Maximum atoms per molecule
        hidden_dim_z: Edge track hidden dimension
        hidden_dim_s: Node output dimension
        num_layers: Number of PairMixer blocks
        num_heads: Number of attention heads (for node aggregation)
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        t_embed_dim: Timestep embedding dimension
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        max_atoms: int,
        hidden_dim_z: int = 128,
        hidden_dim_s: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        t_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.max_atoms = max_atoms
        self.hidden_dim_z = hidden_dim_z
        self.hidden_dim_s = hidden_dim_s
        self.num_layers = num_layers

        if t_embed_dim is None:
            t_embed_dim = hidden_dim_z

        # Input projections
        self.node_embed = nn.Linear(num_atom_types, hidden_dim_s)
        self.edge_embed = nn.Linear(num_bond_types, hidden_dim_z)

        # Positional embeddings for edges (row/column factorized)
        self.edge_pos_embed_row = nn.Parameter(
            torch.randn(1, max_atoms, 1, hidden_dim_z // 2) * 0.02
        )
        self.edge_pos_embed_col = nn.Parameter(
            torch.randn(1, 1, max_atoms, hidden_dim_z // 2) * 0.02
        )

        # Node positional embedding (for aggregation queries)
        self.node_pos_embed = nn.Parameter(
            torch.randn(1, max_atoms, hidden_dim_s) * 0.02
        )

        # Timestep embedding network
        self.t_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, hidden_dim_z * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim_z * 4, hidden_dim_z),
        )

        # PairMixer blocks (edge-only)
        self.blocks = nn.ModuleList([
            PairMixerBlock(
                d_z=hidden_dim_z,
                cond_dim=hidden_dim_z,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Node aggregation from edges
        self.node_aggregation = NodeFromEdgeAttention(
            d_s=hidden_dim_s,
            d_z=hidden_dim_z,
            num_heads=num_heads,
            dropout=dropout,
        )

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
        t_emb = self.t_embed(t)  # [B, hidden_dim_z]

        # Project inputs to hidden dimensions
        s_init = self.node_embed(x)  # [B, N, hidden_dim_s]
        z = self.edge_embed(adj)  # [B, N, N, hidden_dim_z]

        # Add positional embeddings to nodes (for aggregation queries later)
        s_init = s_init + self.node_pos_embed

        # Add positional embeddings to edges
        edge_pos = torch.cat([
            self.edge_pos_embed_row.expand(B, -1, N, -1),
            self.edge_pos_embed_col.expand(B, N, -1, -1),
        ], dim=-1)
        z = z + edge_pos

        # Apply PairMixer blocks (edge-only)
        for block in self.blocks:
            z = block(z, t_emb, mask)

        # Aggregate edge features to node features via attention
        s = self.node_aggregation(z, s_init, mask)
        # Add residual from initial node embedding
        s = s + s_init

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


def create_pairmixer_from_config(config: dict) -> PairMixerFlow:
    """Create PairMixerFlow model from configuration dictionary.

    Args:
        config: Configuration dictionary with 'model' and 'graph' sections

    Returns:
        Initialized PairMixerFlow model
    """
    graph_config = config.get("graph", {})
    model_config = config.get("model", {})

    return PairMixerFlow(
        num_atom_types=len(graph_config.get("atom_types", ["C", "N", "O", "F", "H"])),
        num_bond_types=len(
            graph_config.get("bond_types", ["none", "single", "double", "triple", "aromatic"])
        ),
        max_atoms=graph_config.get("max_atoms", 9),
        hidden_dim_z=model_config.get("hidden_dim_z", model_config.get("hidden_dim", 128)),
        hidden_dim_s=model_config.get("hidden_dim_s", model_config.get("hidden_dim", 128)),
        num_layers=model_config.get("num_layers", 6),
        num_heads=model_config.get("num_heads", 4),
        mlp_ratio=model_config.get("mlp_ratio", 4.0),
        dropout=model_config.get("dropout", 0.1),
        t_embed_dim=model_config.get("t_embed_dim"),
    )


def test_pairmixer():
    """Test PairMixer components."""
    print("Testing PairMixer components...")

    batch_size = 4
    max_atoms = 9
    num_atom_types = 5
    num_bond_types = 5
    hidden_dim_z = 64
    hidden_dim_s = 64
    num_layers = 2
    num_heads = 4
    cond_dim = 64

    x = torch.randn(batch_size, max_atoms, num_atom_types)
    adj = torch.randn(batch_size, max_atoms, max_atoms, num_bond_types)
    z = torch.randn(batch_size, max_atoms, max_atoms, hidden_dim_z)
    s = torch.randn(batch_size, max_atoms, hidden_dim_s)
    t = torch.rand(batch_size)
    t_emb = torch.randn(batch_size, cond_dim)
    mask = torch.ones(batch_size, max_atoms).bool()
    mask[:, 7:] = False

    # Test NodeFromEdgeAttention
    print("\n1. NodeFromEdgeAttention:")
    node_agg = NodeFromEdgeAttention(hidden_dim_s, hidden_dim_z, num_heads)
    out = node_agg(z, s, mask)
    print(f"   Input z: {z.shape}, s_query: {s.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == s.shape
    print("   PASSED")

    # Test PairMixerBlock
    print("\n2. PairMixerBlock:")
    block = PairMixerBlock(hidden_dim_z, cond_dim)
    z_out = block(z, t_emb, mask)
    print(f"   Input z: {z.shape}")
    print(f"   Output: {z_out.shape}")
    assert z_out.shape == z.shape
    print("   PASSED")

    # Test PairMixerFlow
    print("\n3. PairMixerFlow:")
    model = PairMixerFlow(
        num_atom_types=num_atom_types,
        num_bond_types=num_bond_types,
        max_atoms=max_atoms,
        hidden_dim_z=hidden_dim_z,
        hidden_dim_s=hidden_dim_s,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    v_x, v_adj = model(x, adj, t, mask)
    print(f"   Input x: {x.shape}, adj: {adj.shape}")
    print(f"   Output v_x: {v_x.shape}, v_adj: {v_adj.shape}")
    print(f"   Parameters: {model.count_parameters():,}")
    assert v_x.shape == (batch_size, max_atoms, num_atom_types)
    assert v_adj.shape == (batch_size, max_atoms, max_atoms, num_bond_types)
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
            "hidden_dim_z": 64,
            "hidden_dim_s": 64,
            "num_layers": 2,
            "num_heads": 4,
        },
    }
    model_cfg = create_pairmixer_from_config(config)
    v_x, v_adj = model_cfg(x, adj, t, mask)
    print(f"   Created from config successfully")
    print(f"   Parameters: {model_cfg.count_parameters():,}")
    print("   PASSED")

    # Compare parameter count with PairFormer
    print("\n5. Parameter comparison:")
    from src.models.pairformer_flow import PairFormerFlow

    pairformer = PairFormerFlow(
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

    pairformer_no_attn = PairFormerFlow(
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

    pairmixer = PairMixerFlow(
        num_atom_types=num_atom_types,
        num_bond_types=num_bond_types,
        max_atoms=max_atoms,
        hidden_dim_z=hidden_dim_z,
        hidden_dim_s=hidden_dim_s,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    print(f"   PairFormer (full):         {pairformer.count_parameters():,} params")
    print(f"   PairFormer (no tri-attn):  {pairformer_no_attn.count_parameters():,} params")
    print(f"   PairMixer:                 {pairmixer.count_parameters():,} params")

    print("\nAll PairMixer tests passed!")


if __name__ == "__main__":
    test_pairmixer()
