"""Positional encoding implementations for molecular graph transformers.

Implements various positional encoding strategies:
- Learnable: Separate learnable embeddings for nodes and edges
- Sinusoidal: Fixed sine/cosine functions
- RoPE: Rotary position embeddings applied to Q/K
- Relative Bias: Learnable bias added to attention scores
- Graph Distance: Shortest path distance encoding
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalSequenceEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding for sequences.

    Uses the original Transformer positional encoding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        hidden_dim: Embedding dimension
        max_len: Maximum sequence length
    """

    def __init__(self, hidden_dim: int, max_len: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Pre-compute positional encodings
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, hidden_dim]

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding for sequence.

        Args:
            seq_len: Sequence length

        Returns:
            Positional encoding [1, seq_len, hidden_dim]
        """
        return self.pe[:, :seq_len]


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes relative position through rotation of query/key vectors.
    The dot product q_m · k_n depends only on (m - n), providing
    translation-invariant relative position encoding.

    Reference: RoFormer (Su et al., 2021)

    Args:
        dim: Dimension to apply RoPE (typically head_dim)
        base: Base for frequency computation
        max_len: Maximum sequence length
    """

    def __init__(self, dim: int, base: float = 10000.0, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_len = max_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute sin/cos cache
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim/2]

        # Create [seq_len, dim] by repeating each frequency
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]

        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            positions: Optional position indices [seq_len] (default: 0, 1, 2, ...)

        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]

        if positions is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[positions].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[positions].unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


class RelativePositionBias(nn.Module):
    """Learnable relative position bias (T5-style).

    Adds a learned bias to attention scores based on relative distance:
    attention_scores = QK^T / sqrt(d) + bias[i-j]

    Args:
        num_heads: Number of attention heads
        max_distance: Maximum relative distance to embed
        bidirectional: If True, use different embeddings for +/- distances
    """

    def __init__(
        self,
        num_heads: int,
        max_distance: int = 32,
        bidirectional: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Number of unique relative positions
        if bidirectional:
            num_buckets = 2 * max_distance + 1
        else:
            num_buckets = max_distance + 1

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _compute_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Map relative positions to bucket indices."""
        if self.bidirectional:
            # Shift to [0, 2*max_distance]
            relative_position = torch.clamp(
                relative_position, -self.max_distance, self.max_distance
            )
            return relative_position + self.max_distance
        else:
            # Only positive distances
            return torch.clamp(relative_position.abs(), 0, self.max_distance)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute relative position bias matrix.

        Args:
            seq_len: Sequence length
            device: Device for computation

        Returns:
            Bias tensor [num_heads, seq_len, seq_len]
        """
        context_position = torch.arange(seq_len, dtype=torch.long, device=device)
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device)

        # Compute relative positions [seq_len, seq_len]
        relative_position = memory_position.unsqueeze(0) - context_position.unsqueeze(1)

        # Map to buckets
        buckets = self._compute_bucket(relative_position)

        # Get embeddings [seq_len, seq_len, num_heads]
        bias = self.relative_attention_bias(buckets)

        # Transpose to [num_heads, seq_len, seq_len]
        return bias.permute(2, 0, 1)


class GraphDistanceEncoding(nn.Module):
    """Graph distance encoding (Graphormer-style).

    Computes shortest path distances between nodes and embeds them
    as attention biases. This captures the molecular graph structure.

    Args:
        num_heads: Number of attention heads
        max_distance: Maximum graph distance to embed
        num_atom_types: Number of atom types (for optional node centrality)
    """

    def __init__(
        self,
        num_heads: int,
        max_distance: int = 8,
        num_atom_types: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

        # Distance embedding: distances 0 to max_distance, plus one for "no path"
        self.distance_embed = nn.Embedding(max_distance + 2, num_heads)

        # Optional node degree embedding
        if num_atom_types is not None:
            self.degree_embed = nn.Embedding(max_distance + 1, num_heads)
        else:
            self.degree_embed = None

    def compute_shortest_paths(self, adj: torch.Tensor) -> torch.Tensor:
        """Compute shortest path distances using Floyd-Warshall.

        Args:
            adj: Binary adjacency matrix [batch, N, N] (1 = connected, 0 = not)

        Returns:
            Distance matrix [batch, N, N] (inf for no path)
        """
        B, N, _ = adj.shape
        device = adj.device

        # Initialize distance matrix
        # 0 for self-loops, 1 for direct edges, inf for no connection
        dist = torch.full((B, N, N), float('inf'), device=device)

        # Self-loops have distance 0
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        dist = dist.masked_fill(eye.bool(), 0.0)

        # Direct edges have distance 1
        # adj should be binary (0 or 1)
        dist = dist.masked_fill(adj > 0.5, 1.0)

        # Floyd-Warshall
        for k in range(N):
            dist = torch.minimum(dist, dist[:, :, k:k+1] + dist[:, k:k+1, :])

        return dist

    def forward(
        self,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute graph distance bias for attention.

        Args:
            adj: Adjacency matrix [batch, N, N] or [batch, N, N, E]
            mask: Node validity mask [batch, N]

        Returns:
            Distance bias [batch, num_heads, N, N]
        """
        # Handle multi-edge adjacency
        if adj.dim() == 4:
            # Sum over edge types, excluding "no bond" (index 0)
            adj_binary = (adj[..., 1:].sum(dim=-1) > 0.5).float()
        else:
            adj_binary = (adj > 0.5).float()

        B, N, _ = adj_binary.shape

        # Compute shortest paths
        distances = self.compute_shortest_paths(adj_binary)  # [B, N, N]

        # Clamp and handle inf
        distances_clamped = torch.clamp(distances, 0, self.max_distance)
        no_path_mask = distances == float('inf')

        # Convert to long for embedding lookup
        # Use max_distance + 1 for "no path"
        distance_indices = distances_clamped.long()
        distance_indices = distance_indices.masked_fill(no_path_mask, self.max_distance + 1)

        # Apply mask if provided
        if mask is not None:
            invalid_mask = ~(mask.unsqueeze(2) & mask.unsqueeze(1))
            distance_indices = distance_indices.masked_fill(invalid_mask, self.max_distance + 1)

        # Get embeddings [B, N, N, num_heads]
        bias = self.distance_embed(distance_indices)

        # Transpose to [B, num_heads, N, N]
        return bias.permute(0, 3, 1, 2)

    def get_node_centrality(
        self,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute node degree centrality embedding.

        Args:
            adj: Adjacency matrix [batch, N, N] or [batch, N, N, E]
            mask: Node validity mask [batch, N]

        Returns:
            Degree embedding [batch, N, num_heads]
        """
        if self.degree_embed is None:
            raise ValueError("Node centrality requires num_atom_types in init")

        # Handle multi-edge adjacency
        if adj.dim() == 4:
            adj_binary = (adj[..., 1:].sum(dim=-1) > 0.5).float()
        else:
            adj_binary = (adj > 0.5).float()

        # Compute degrees
        degrees = adj_binary.sum(dim=-1)  # [B, N]

        # Clamp and convert to long
        degrees = torch.clamp(degrees, 0, self.max_distance).long()

        # Apply mask
        if mask is not None:
            degrees = degrees.masked_fill(~mask, 0)

        return self.degree_embed(degrees)  # [B, N, num_heads]


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embeddings for nodes and edges.

    This is the default positional encoding used in the original GraphDiT.

    Args:
        max_atoms: Maximum number of atoms
        hidden_dim: Hidden dimension
        init_std: Standard deviation for initialization
    """

    def __init__(
        self,
        max_atoms: int,
        hidden_dim: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim

        # Node positional embeddings
        self.node_pos_embed = nn.Parameter(
            torch.randn(1, max_atoms, hidden_dim) * init_std
        )

        # Edge positional embeddings
        self.edge_pos_embed = nn.Parameter(
            torch.randn(1, max_atoms * max_atoms, hidden_dim) * init_std
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional embeddings to node and edge features.

        Args:
            node_feat: Node features [batch, N, hidden_dim]
            edge_feat: Edge features [batch, N*N, hidden_dim]

        Returns:
            (node_feat + PE, edge_feat + PE)
        """
        return node_feat + self.node_pos_embed, edge_feat + self.edge_pos_embed


def create_positional_encoding(
    pe_config: dict,
    hidden_dim: int,
    num_heads: int,
    max_atoms: int,
) -> nn.Module:
    """Factory function to create positional encoding from config.

    Args:
        pe_config: Configuration dict with 'type' and type-specific params
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        max_atoms: Maximum atoms per molecule

    Returns:
        Positional encoding module
    """
    pe_type = pe_config.get('type', 'learnable')

    if pe_type == 'learnable':
        return LearnablePositionalEmbedding(
            max_atoms=max_atoms,
            hidden_dim=hidden_dim,
            init_std=pe_config.get('init_std', 0.02),
        )

    elif pe_type == 'sinusoidal':
        return SinusoidalSequenceEmbedding(
            hidden_dim=hidden_dim,
            max_len=max_atoms + max_atoms * max_atoms + 10,
        )

    elif pe_type == 'rope':
        return RotaryPositionalEmbedding(
            dim=hidden_dim // num_heads,
            base=pe_config.get('base', 10000.0),
            max_len=max_atoms + max_atoms * max_atoms + 10,
        )

    elif pe_type == 'relative_bias':
        return RelativePositionBias(
            num_heads=num_heads,
            max_distance=pe_config.get('max_distance', 32),
            bidirectional=pe_config.get('bidirectional', True),
        )

    elif pe_type == 'graph_distance':
        return GraphDistanceEncoding(
            num_heads=num_heads,
            max_distance=pe_config.get('max_graph_distance', 8),
            num_atom_types=pe_config.get('num_atom_types'),
        )

    else:
        raise ValueError(f"Unknown positional encoding type: {pe_type}")


def test_all_pe():
    """Test all positional encoding implementations."""
    print("Testing positional encodings...")

    batch_size = 4
    max_atoms = 9
    hidden_dim = 128
    num_heads = 8
    head_dim = hidden_dim // num_heads

    # Test Sinusoidal
    print("\n1. Sinusoidal PE:")
    sinusoidal = SinusoidalSequenceEmbedding(hidden_dim=hidden_dim, max_len=100)
    pe = sinusoidal(seq_len=50)
    print(f"   Output shape: {pe.shape}")
    assert pe.shape == (1, 50, hidden_dim)
    print("   PASSED")

    # Test RoPE
    print("\n2. RoPE:")
    rope = RotaryPositionalEmbedding(dim=head_dim, max_len=100)
    q = torch.randn(batch_size, num_heads, 50, head_dim)
    k = torch.randn(batch_size, num_heads, 50, head_dim)
    q_rot, k_rot = rope(q, k)
    print(f"   Q shape: {q_rot.shape}, K shape: {k_rot.shape}")
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    print("   PASSED")

    # Test Relative Bias
    print("\n3. Relative Position Bias:")
    rel_bias = RelativePositionBias(num_heads=num_heads, max_distance=16)
    bias = rel_bias(seq_len=50, device=torch.device('cpu'))
    print(f"   Output shape: {bias.shape}")
    assert bias.shape == (num_heads, 50, 50)
    print("   PASSED")

    # Test Graph Distance
    print("\n4. Graph Distance Encoding:")
    graph_dist = GraphDistanceEncoding(num_heads=num_heads, max_distance=8)
    adj = torch.randint(0, 2, (batch_size, max_atoms, max_atoms)).float()
    adj = (adj + adj.transpose(-1, -2)) / 2  # Make symmetric
    adj.diagonal(dim1=-2, dim2=-1).fill_(0)  # No self-loops
    mask = torch.ones(batch_size, max_atoms).bool()
    mask[:, 7:] = False  # Some atoms invalid

    bias = graph_dist(adj, mask)
    print(f"   Output shape: {bias.shape}")
    assert bias.shape == (batch_size, num_heads, max_atoms, max_atoms)
    print("   PASSED")

    # Test Learnable
    print("\n5. Learnable PE:")
    learnable = LearnablePositionalEmbedding(max_atoms=max_atoms, hidden_dim=hidden_dim)
    node_feat = torch.randn(batch_size, max_atoms, hidden_dim)
    edge_feat = torch.randn(batch_size, max_atoms * max_atoms, hidden_dim)
    node_out, edge_out = learnable(node_feat, edge_feat)
    print(f"   Node shape: {node_out.shape}, Edge shape: {edge_out.shape}")
    assert node_out.shape == node_feat.shape and edge_out.shape == edge_feat.shape
    print("   PASSED")

    print("\nAll positional encoding tests passed!")


if __name__ == "__main__":
    test_all_pe()
