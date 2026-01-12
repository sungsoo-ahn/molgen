"""DiT (Diffusion Transformer) architecture for molecular graph generation.

Implements a transformer-based velocity field predictor for flow matching,
adapted for 2D molecular graphs with atom types and bond adjacency matrices.

Supports multiple positional encoding types:
- learnable: Separate learnable embeddings for nodes and edges (default)
- sinusoidal: Fixed sine/cosine functions
- relative_bias: Learnable bias added to attention scores
- graph_distance: Shortest path distance encoding
- rope: Rotary position embeddings (applied in attention)
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional import (
    LearnablePositionalEmbedding,
    SinusoidalSequenceEmbedding,
    RelativePositionBias,
    GraphDistanceEncoding,
    RotaryPositionalEmbedding,
)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps.

    Maps scalar timesteps to high-dimensional embeddings using sine/cosine
    functions at different frequencies.

    Args:
        dim: Output embedding dimension
        max_period: Maximum period for sinusoidal functions
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timestep tensor [batch_size] with values in [0, 1]

        Returns:
            Embeddings [batch_size, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )

        # Scale timesteps and compute sinusoidal embeddings
        args = t[:, None].float() * freqs[None, :]  # [batch, half_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding.

    Applies LayerNorm followed by learned scale and shift parameters
    derived from the conditioning signal.

    Args:
        hidden_dim: Hidden dimension of the input
        cond_dim: Dimension of the conditioning signal
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)  # scale and shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            cond: Conditioning tensor [batch, cond_dim]

        Returns:
            Normalized tensor [batch, seq_len, hidden_dim]
        """
        # Project conditioning to scale and shift
        scale_shift = self.proj(cond)  # [batch, hidden_dim * 2]
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each [batch, hidden_dim]

        # Apply layer norm then modulate
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return x


class DiTBlock(nn.Module):
    """DiT transformer block with adaptive layer normalization.

    Implements a standard transformer block with AdaLN conditioning on timestep:
    x -> AdaLN(x, t) -> Self-Attention -> Residual
    x -> AdaLN(x, t) -> MLP -> Residual

    Supports optional attention bias and RoPE.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        cond_dim: Conditioning dimension (timestep embedding)
        mlp_ratio: MLP hidden dimension multiplier
        dropout: Dropout rate
        use_rope: Whether to use rotary position embeddings
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope

        # AdaLN layers
        self.norm1 = AdaLN(hidden_dim, cond_dim)
        self.norm2 = AdaLN(hidden_dim, cond_dim)

        # Self-attention (manual implementation for RoPE and bias support)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # MLP
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            cond: Conditioning tensor [batch, cond_dim]
            attn_bias: Attention bias [batch, num_heads, seq_len, seq_len] or
                       [num_heads, seq_len, seq_len]
            key_padding_mask: Padding mask [batch, seq_len] - True for padding
            rope: Optional RoPE module for rotary embeddings

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        B, L, D = x.shape

        # Self-attention with AdaLN
        normed = self.norm1(x, cond)

        # Q, K, V projections
        q = self.q_proj(normed).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rope is not None:
            q, k = rope(q, k)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]

        # Add attention bias if provided
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                # [H, L, L] -> [1, H, L, L]
                attn_bias = attn_bias.unsqueeze(0)
            scores = scores + attn_bias

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] where True = padding
            # Expand to [B, 1, 1, L] for broadcasting
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [B, H, L, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out

        # MLP with AdaLN
        normed = self.norm2(x, cond)
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x


class GraphDiT(nn.Module):
    """DiT-style transformer for molecular graph generation via flow matching.

    Flattens node features and edge features into a sequence, processes with
    transformer blocks, and outputs velocity predictions for both nodes and edges.

    Architecture:
        Input: (x, adj, t) where x is node features, adj is adjacency, t is timestep
        1. Embed nodes and edges separately
        2. Concatenate into sequence: [node_1, ..., node_N, edge_11, ..., edge_NN]
        3. Add positional embeddings (various types supported)
        4. Process through DiT blocks with timestep conditioning
        5. Split and project back to node/edge velocities

    Positional Encoding Types:
        - learnable: Separate learnable embeddings for nodes and edges (default)
        - sinusoidal: Fixed sine/cosine functions
        - relative_bias: Learnable bias added to attention scores
        - graph_distance: Shortest path distance encoding
        - rope: Rotary position embeddings

    Args:
        num_atom_types: Number of atom types
        num_bond_types: Number of bond types (including no-bond)
        max_atoms: Maximum atoms per molecule
        hidden_dim: Transformer hidden dimension
        num_layers: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        t_embed_dim: Timestep embedding dimension (defaults to hidden_dim)
        pe_type: Positional encoding type (default: 'learnable')
        pe_config: Additional config for positional encoding
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        max_atoms: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        t_embed_dim: Optional[int] = None,
        pe_type: str = 'learnable',
        pe_config: Optional[dict] = None,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pe_type = pe_type

        if t_embed_dim is None:
            t_embed_dim = hidden_dim

        if pe_config is None:
            pe_config = {}

        # Sequence length: nodes + edges (N + N*N)
        self.num_nodes = max_atoms
        self.num_edges = max_atoms * max_atoms
        self.seq_len = self.num_nodes + self.num_edges

        # Input projections
        self.node_embed = nn.Linear(num_atom_types, hidden_dim)
        self.edge_embed = nn.Linear(num_bond_types, hidden_dim)

        # Positional encoding setup based on type
        self._setup_positional_encoding(pe_type, pe_config)

        # Timestep embedding network
        self.t_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # DiT blocks
        use_rope = pe_type == 'rope'
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                cond_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_rope=use_rope,
            )
            for _ in range(num_layers)
        ])

        # Final normalization and projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_node = nn.Linear(hidden_dim, num_atom_types)
        self.out_edge = nn.Linear(hidden_dim, num_bond_types)

        # Initialize output layers to zero for better training stability
        nn.init.zeros_(self.out_node.weight)
        nn.init.zeros_(self.out_node.bias)
        nn.init.zeros_(self.out_edge.weight)
        nn.init.zeros_(self.out_edge.bias)

    def _setup_positional_encoding(self, pe_type: str, pe_config: dict):
        """Setup positional encoding based on type."""
        if pe_type == 'learnable':
            # Learnable positional embeddings (default)
            self.pos_embed = LearnablePositionalEmbedding(
                max_atoms=self.max_atoms,
                hidden_dim=self.hidden_dim,
                init_std=pe_config.get('init_std', 0.02),
            )
            self.rope = None
            self.relative_bias = None
            self.graph_distance = None

        elif pe_type == 'sinusoidal':
            # Fixed sinusoidal encoding
            self.pos_embed = SinusoidalSequenceEmbedding(
                hidden_dim=self.hidden_dim,
                max_len=self.seq_len + 10,
            )
            self.rope = None
            self.relative_bias = None
            self.graph_distance = None

        elif pe_type == 'rope':
            # Rotary position embeddings (applied in attention)
            self.pos_embed = None
            self.rope = RotaryPositionalEmbedding(
                dim=self.hidden_dim // self.num_heads,
                base=pe_config.get('base', 10000.0),
                max_len=self.seq_len + 10,
            )
            self.relative_bias = None
            self.graph_distance = None

        elif pe_type == 'relative_bias':
            # Relative position bias added to attention
            self.pos_embed = None
            self.rope = None
            self.relative_bias = RelativePositionBias(
                num_heads=self.num_heads,
                max_distance=pe_config.get('max_distance', 32),
                bidirectional=pe_config.get('bidirectional', True),
            )
            self.graph_distance = None

        elif pe_type == 'graph_distance':
            # Graph distance encoding
            self.pos_embed = None
            self.rope = None
            self.relative_bias = None
            self.graph_distance = GraphDistanceEncoding(
                num_heads=self.num_heads,
                max_distance=pe_config.get('max_graph_distance', 8),
            )

        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}")

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
        t_emb = self.t_embed(t)  # [B, hidden_dim]

        # Project inputs to hidden dimension
        node_feat = self.node_embed(x)  # [B, N, hidden_dim]
        edge_feat = self.edge_embed(adj.view(B, N * N, E))  # [B, N*N, hidden_dim]

        # Apply positional embeddings based on type
        if self.pe_type == 'learnable':
            node_feat, edge_feat = self.pos_embed(node_feat, edge_feat)

        elif self.pe_type == 'sinusoidal':
            # Add sinusoidal positional encoding to concatenated sequence
            pe = self.pos_embed(self.seq_len)
            node_feat = node_feat + pe[:, :N]
            edge_feat = edge_feat + pe[:, N:self.seq_len]

        # For rope, relative_bias, graph_distance: no additive PE,
        # bias is computed and passed to attention

        # Concatenate into sequence: [nodes, edges]
        seq = torch.cat([node_feat, edge_feat], dim=1)  # [B, N + N*N, hidden_dim]

        # Create key padding mask for attention
        # True = ignore, False = attend
        key_padding_mask = self._create_key_padding_mask(mask)

        # Compute attention bias if needed
        attn_bias = self._compute_attention_bias(adj, mask, seq.device)

        # Apply DiT blocks
        for block in self.blocks:
            seq = block(
                seq, t_emb,
                attn_bias=attn_bias,
                key_padding_mask=key_padding_mask,
                rope=self.rope,
            )

        # Final normalization
        seq = self.final_norm(seq)

        # Split back into nodes and edges
        node_out = seq[:, :N]  # [B, N, hidden_dim]
        edge_out = seq[:, N:]  # [B, N*N, hidden_dim]

        # Project to output dimensions
        v_x = self.out_node(node_out)  # [B, N, num_atom_types]
        v_adj = self.out_edge(edge_out).view(B, N, N, E)  # [B, N, N, num_bond_types]

        return v_x, v_adj

    def _compute_attention_bias(
        self,
        adj: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Compute attention bias based on positional encoding type.

        Args:
            adj: Adjacency features [batch, N, N, E]
            mask: Node mask [batch, N]
            device: Device for computation

        Returns:
            Attention bias [batch, num_heads, seq_len, seq_len] or None
        """
        if self.pe_type == 'relative_bias':
            # Relative position bias: [num_heads, seq_len, seq_len]
            return self.relative_bias(self.seq_len, device)

        elif self.pe_type == 'graph_distance':
            # Graph distance bias for node-node attention
            # Compute node distance bias: [B, num_heads, N, N]
            node_bias = self.graph_distance(adj, mask)

            B = adj.shape[0]
            # Expand to full sequence: [B, num_heads, seq_len, seq_len]
            full_bias = torch.zeros(
                B, self.num_heads, self.seq_len, self.seq_len,
                device=device
            )

            # Fill node-node block
            full_bias[:, :, :self.num_nodes, :self.num_nodes] = node_bias

            # For node-edge and edge-edge blocks, we could add more sophisticated
            # biases, but for now just use zero (no additional bias)
            # Future work: add edge distance biases based on endpoint distances

            return full_bias

        return None

    def _create_key_padding_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create key padding mask for attention.

        Args:
            mask: Node mask [batch, max_atoms] - True for valid atoms

        Returns:
            Key padding mask [batch, seq_len] - True for padding (to ignore)
        """
        B, N = mask.shape

        # Node padding: ~mask (True where padding)
        node_padding = ~mask  # [B, N]

        # Edge padding: edge (i,j) is padding if either node i or j is padding
        # Create edge mask: valid if both endpoints are valid
        edge_valid = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        edge_padding = ~edge_valid.view(B, N * N)  # [B, N*N]

        # Concatenate
        return torch.cat([node_padding, edge_padding], dim=1)  # [B, N + N*N]

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_from_config(config: dict) -> GraphDiT:
    """Create GraphDiT model from configuration dictionary.

    Args:
        config: Configuration dictionary with 'model' and 'graph' sections

    Returns:
        Initialized GraphDiT model
    """
    graph_config = config.get("graph", {})
    model_config = config.get("model", {})

    # Extract positional encoding config
    pe_config = model_config.get("positional_encoding", {})
    pe_type = pe_config.get("type", "learnable")

    return GraphDiT(
        num_atom_types=len(graph_config.get("atom_types", ["C", "N", "O", "F", "H"])),
        num_bond_types=len(
            graph_config.get("bond_types", ["none", "single", "double", "triple", "aromatic"])
        ),
        max_atoms=graph_config.get("max_atoms", 9),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 8),
        num_heads=model_config.get("num_heads", 8),
        mlp_ratio=model_config.get("mlp_ratio", 4.0),
        dropout=model_config.get("dropout", 0.1),
        t_embed_dim=model_config.get("t_embed_dim"),
        pe_type=pe_type,
        pe_config=pe_config,
    )
