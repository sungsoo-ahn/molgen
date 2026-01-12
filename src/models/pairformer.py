"""PairFormer architecture for molecular graph generation.

Implements a dual-track transformer with separate node and edge representations,
inspired by AlphaFold2's pair representation updates. Key components:

- AttentionPairBias: Node attention with pair representations as bias
- TriangleMultiplication: Edge updates via triangle interactions
- TriangleAttention: Attention along different axes of pair representation
- PairFormerBlock: Complete block combining all operations

Reference: AlphaFold2, BoltzGen (https://github.com/HannesStark/boltzgen)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNPair(nn.Module):
    """Adaptive Layer Normalization for pair representations.

    Applies LayerNorm followed by learned scale and shift from conditioning.

    Args:
        dim: Feature dimension
        cond_dim: Conditioning dimension
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor [batch, ..., dim]
            cond: Conditioning tensor [batch, cond_dim]

        Returns:
            Normalized tensor with same shape as x
        """
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Expand scale/shift to match x dimensions
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        x = self.norm(x)
        return x * (1 + scale) + shift


class AttentionPairBias(nn.Module):
    """Multi-head attention with pair representations as bias.

    Incorporates edge information directly into attention scores:
    attention = softmax((QK^T + pair_bias) / sqrt(d))

    Args:
        d_s: Node feature dimension
        d_z: Pair feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_s: int,
        d_z: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_s // num_heads

        assert d_s % num_heads == 0, f"d_s ({d_s}) must be divisible by num_heads ({num_heads})"

        # Q, K, V projections for node features
        self.q_proj = nn.Linear(d_s, d_s)
        self.k_proj = nn.Linear(d_s, d_s)
        self.v_proj = nn.Linear(d_s, d_s)

        # Pair bias projection: z[i,j] -> [num_heads]
        self.pair_bias_norm = nn.LayerNorm(d_z)
        self.pair_bias_proj = nn.Linear(d_z, num_heads, bias=False)

        # Output projection with gating
        self.gate_proj = nn.Linear(d_s, d_s)
        self.out_proj = nn.Linear(d_s, d_s)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.zeros_(self.pair_bias_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pair bias.

        Args:
            s: Node features [batch, N, d_s]
            z: Pair features [batch, N, N, d_z]
            mask: Node validity mask [batch, N] - True for valid

        Returns:
            Updated node features [batch, N, d_s]
        """
        B, N, _ = s.shape

        # Q, K, V projections
        q = self.q_proj(s).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(s).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(s).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, N, N]

        # Add pair bias
        z_normed = self.pair_bias_norm(z)
        pair_bias = self.pair_bias_proj(z_normed)  # [B, N, N, num_heads]
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # [B, num_heads, N, N]
        scores = scores + pair_bias

        # Apply mask
        if mask is not None:
            # Create attention mask: [B, 1, 1, N]
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_s)

        # Gated output
        gate = torch.sigmoid(self.gate_proj(s))
        out = gate * self.out_proj(out)

        return out


class TriangleMultiplication(nn.Module):
    """Triangle multiplication for pair representation updates.

    Updates pair representations via triangle interactions:
    - Outgoing: z[i,j] += sum_k g(z[i,k]) * g(z[k,j])
    - Incoming: z[i,j] += sum_k g(z[k,i]) * g(z[k,j])

    This captures multi-hop information flow in the molecular graph.

    Args:
        d_z: Pair feature dimension
        d_hidden: Hidden dimension for gating
        outgoing: If True, use outgoing direction; else incoming
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_z: int,
        d_hidden: Optional[int] = None,
        outgoing: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_z = d_z
        self.d_hidden = d_hidden or d_z
        self.outgoing = outgoing

        # Layer norm for input
        self.norm = nn.LayerNorm(d_z)

        # Left and right projections with gating
        self.left_proj = nn.Linear(d_z, self.d_hidden)
        self.left_gate = nn.Linear(d_z, self.d_hidden)
        self.right_proj = nn.Linear(d_z, self.d_hidden)
        self.right_gate = nn.Linear(d_z, self.d_hidden)

        # Output projection with gating
        self.out_norm = nn.LayerNorm(self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_z)
        self.out_gate = nn.Linear(d_z, d_z)

        self.dropout = nn.Dropout(dropout)

        # Initialize gates to zero (start with identity)
        nn.init.zeros_(self.left_gate.bias)
        nn.init.zeros_(self.right_gate.bias)
        nn.init.zeros_(self.out_gate.bias)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Pair features [batch, N, N, d_z]
            mask: Node validity mask [batch, N] - True for valid

        Returns:
            Updated pair features [batch, N, N, d_z]
        """
        B, N, _, _ = z.shape

        # Normalize input
        z_in = self.norm(z)

        # Project with gating
        left = self.left_proj(z_in) * torch.sigmoid(self.left_gate(z_in))
        right = self.right_proj(z_in) * torch.sigmoid(self.right_gate(z_in))

        # Apply mask to inputs
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
            pair_mask = pair_mask.unsqueeze(-1)  # [B, N, N, 1]
            left = left * pair_mask
            right = right * pair_mask

        # Triangle multiplication
        if self.outgoing:
            # z_out[i,j] = sum_k left[i,k] * right[k,j]
            # left: [B, N, N, d_hidden] -> [B, i, k, d_hidden]
            # right: [B, N, N, d_hidden] -> [B, k, j, d_hidden]
            out = torch.einsum('bikd,bkjd->bijd', left, right)
        else:
            # z_out[i,j] = sum_k left[k,i] * right[k,j]
            # left: [B, k, i, d_hidden]
            # right: [B, k, j, d_hidden]
            out = torch.einsum('bkid,bkjd->bijd', left, right)

        # Output projection with gating
        out = self.out_norm(out)
        out = self.out_proj(out)
        gate = torch.sigmoid(self.out_gate(z))
        out = gate * self.dropout(out)

        return out


class TriangleAttention(nn.Module):
    """Attention over pair representations along different axes.

    For starting node attention: attend along j for each fixed i
    For ending node attention: attend along i for each fixed j

    Args:
        d_z: Pair feature dimension
        num_heads: Number of attention heads
        starting: If True, attend along j (starting node); else along i
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_z: int,
        num_heads: int = 4,
        starting: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_z // num_heads
        self.starting = starting

        assert d_z % num_heads == 0

        # Layer norm
        self.norm = nn.LayerNorm(d_z)

        # Q, K, V projections
        self.q_proj = nn.Linear(d_z, d_z)
        self.k_proj = nn.Linear(d_z, d_z)
        self.v_proj = nn.Linear(d_z, d_z)

        # Bias from other dimension
        self.bias_proj = nn.Linear(d_z, num_heads, bias=False)

        # Output projection with gating
        self.gate_proj = nn.Linear(d_z, d_z)
        self.out_proj = nn.Linear(d_z, d_z)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Pair features [batch, N, N, d_z]
            mask: Node validity mask [batch, N] - True for valid

        Returns:
            Updated pair features [batch, N, N, d_z]
        """
        B, N, N2, _ = z.shape
        assert N == N2

        z_in = self.norm(z)

        if self.starting:
            # For each row i, attend over columns j
            # Reshape to process each row independently
            z_flat = z_in.reshape(B * N, N, self.d_z)
            z_orig_flat = z.reshape(B * N, N, self.d_z)

            # Create bias from column information
            # Use mean over rows as bias source
            bias_input = z_in.mean(dim=1)  # [B, N, d_z]
            bias = self.bias_proj(bias_input)  # [B, N, num_heads]
            bias = bias.unsqueeze(1).expand(-1, N, -1, -1).contiguous()  # [B, N, N, num_heads]
            bias = bias.reshape(B * N, N, self.num_heads)

            # Mask for columns
            if mask is not None:
                col_mask = mask.unsqueeze(1).expand(-1, N, -1).contiguous()  # [B, N, N]
                col_mask = col_mask.reshape(B * N, N)
            else:
                col_mask = None

        else:
            # For each column j, attend over rows i
            # Transpose and reshape
            z_flat = z_in.transpose(1, 2).contiguous().reshape(B * N, N, self.d_z)
            z_orig_flat = z.transpose(1, 2).contiguous().reshape(B * N, N, self.d_z)

            # Use mean over columns as bias source
            bias_input = z_in.mean(dim=2)  # [B, N, d_z]
            bias = self.bias_proj(bias_input)  # [B, N, num_heads]
            bias = bias.unsqueeze(1).expand(-1, N, -1, -1).contiguous()  # [B, N, N, num_heads]
            bias = bias.reshape(B * N, N, self.num_heads)

            # Mask for rows
            if mask is not None:
                row_mask = mask.unsqueeze(2).expand(-1, -1, N).contiguous()  # [B, N, N]
                row_mask = row_mask.transpose(1, 2).contiguous().reshape(B * N, N)
            else:
                row_mask = None
            col_mask = row_mask

        # Q, K, V projections
        q = self.q_proj(z_flat).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z_flat).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z_flat).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B*N, H, N, N]

        # Add bias
        bias = bias.permute(0, 2, 1).unsqueeze(-1)  # [B*N, num_heads, N, 1]
        scores = scores + bias

        # Apply mask
        if col_mask is not None:
            attn_mask = ~col_mask.unsqueeze(1).unsqueeze(2)  # [B*N, 1, 1, N]
            scores = scores.masked_fill(attn_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B * N, N, self.d_z)

        # Gated output
        gate = torch.sigmoid(self.gate_proj(z_orig_flat))
        out = gate * self.out_proj(out)

        # Reshape back
        if self.starting:
            out = out.view(B, N, N, self.d_z)
        else:
            out = out.view(B, N, N, self.d_z).transpose(1, 2).contiguous()

        return out


class PairFormerBlock(nn.Module):
    """Complete PairFormer block with node and pair tracks.

    Processing order:
    1. Node track: AttentionPairBias(s, z) + MLP
    2. Pair track: TriangleMult(out) + TriangleMult(in) + TriangleAttn(start) + TriangleAttn(end) + MLP

    Args:
        d_s: Node feature dimension
        d_z: Pair feature dimension
        cond_dim: Conditioning dimension (for timestep)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        use_triangle_mult: Whether to use triangle multiplication
        use_triangle_attn: Whether to use triangle attention
    """

    def __init__(
        self,
        d_s: int,
        d_z: int,
        cond_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_triangle_mult: bool = True,
        use_triangle_attn: bool = True,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_z = d_z
        self.use_triangle_mult = use_triangle_mult
        self.use_triangle_attn = use_triangle_attn

        # Node track
        self.norm_s1 = AdaLNPair(d_s, cond_dim)
        self.attn_pair_bias = AttentionPairBias(d_s, d_z, num_heads, dropout)
        self.norm_s2 = AdaLNPair(d_s, cond_dim)

        mlp_hidden_s = int(d_s * mlp_ratio)
        self.mlp_s = nn.Sequential(
            nn.Linear(d_s, mlp_hidden_s),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_s, d_s),
            nn.Dropout(dropout),
        )

        # Pair track
        if use_triangle_mult:
            self.norm_z1 = AdaLNPair(d_z, cond_dim)
            self.triangle_mult_out = TriangleMultiplication(d_z, outgoing=True, dropout=dropout)
            self.norm_z2 = AdaLNPair(d_z, cond_dim)
            self.triangle_mult_in = TriangleMultiplication(d_z, outgoing=False, dropout=dropout)

        if use_triangle_attn:
            self.norm_z3 = AdaLNPair(d_z, cond_dim)
            self.triangle_attn_start = TriangleAttention(d_z, num_heads // 2, starting=True, dropout=dropout)
            self.norm_z4 = AdaLNPair(d_z, cond_dim)
            self.triangle_attn_end = TriangleAttention(d_z, num_heads // 2, starting=False, dropout=dropout)

        self.norm_z5 = AdaLNPair(d_z, cond_dim)
        mlp_hidden_z = int(d_z * mlp_ratio)
        self.mlp_z = nn.Sequential(
            nn.Linear(d_z, mlp_hidden_z),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_z, d_z),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            s: Node features [batch, N, d_s]
            z: Pair features [batch, N, N, d_z]
            t_emb: Timestep embedding [batch, cond_dim]
            mask: Node validity mask [batch, N]

        Returns:
            Updated (s, z) tensors
        """
        # Node track
        s_normed = self.norm_s1(s, t_emb)
        s = s + self.attn_pair_bias(s_normed, z, mask)
        s = s + self.mlp_s(self.norm_s2(s, t_emb))

        # Pair track: triangle multiplication
        if self.use_triangle_mult:
            z = z + self.triangle_mult_out(self.norm_z1(z, t_emb), mask)
            z = z + self.triangle_mult_in(self.norm_z2(z, t_emb), mask)

        # Pair track: triangle attention
        if self.use_triangle_attn:
            z = z + self.triangle_attn_start(self.norm_z3(z, t_emb), mask)
            z = z + self.triangle_attn_end(self.norm_z4(z, t_emb), mask)

        # Pair track: MLP
        z = z + self.mlp_z(self.norm_z5(z, t_emb))

        return s, z


def test_pairformer():
    """Test PairFormer components."""
    print("Testing PairFormer components...")

    batch_size = 4
    N = 9
    d_s = 128
    d_z = 64
    num_heads = 8
    cond_dim = 128

    s = torch.randn(batch_size, N, d_s)
    z = torch.randn(batch_size, N, N, d_z)
    t_emb = torch.randn(batch_size, cond_dim)
    mask = torch.ones(batch_size, N).bool()
    mask[:, 7:] = False

    # Test AttentionPairBias
    print("\n1. AttentionPairBias:")
    attn = AttentionPairBias(d_s, d_z, num_heads)
    out = attn(s, z, mask)
    print(f"   Output shape: {out.shape}")
    assert out.shape == s.shape
    print("   PASSED")

    # Test TriangleMultiplication (outgoing)
    print("\n2. TriangleMultiplication (outgoing):")
    tri_out = TriangleMultiplication(d_z, outgoing=True)
    out = tri_out(z, mask)
    print(f"   Output shape: {out.shape}")
    assert out.shape == z.shape
    print("   PASSED")

    # Test TriangleMultiplication (incoming)
    print("\n3. TriangleMultiplication (incoming):")
    tri_in = TriangleMultiplication(d_z, outgoing=False)
    out = tri_in(z, mask)
    print(f"   Output shape: {out.shape}")
    assert out.shape == z.shape
    print("   PASSED")

    # Test TriangleAttention (starting)
    print("\n4. TriangleAttention (starting):")
    tri_attn_s = TriangleAttention(d_z, num_heads // 2, starting=True)
    out = tri_attn_s(z, mask)
    print(f"   Output shape: {out.shape}")
    assert out.shape == z.shape
    print("   PASSED")

    # Test TriangleAttention (ending)
    print("\n5. TriangleAttention (ending):")
    tri_attn_e = TriangleAttention(d_z, num_heads // 2, starting=False)
    out = tri_attn_e(z, mask)
    print(f"   Output shape: {out.shape}")
    assert out.shape == z.shape
    print("   PASSED")

    # Test PairFormerBlock (full)
    print("\n6. PairFormerBlock (full):")
    block = PairFormerBlock(d_s, d_z, cond_dim, num_heads)
    s_out, z_out = block(s, z, t_emb, mask)
    print(f"   s output shape: {s_out.shape}")
    print(f"   z output shape: {z_out.shape}")
    assert s_out.shape == s.shape
    assert z_out.shape == z.shape
    print("   PASSED")

    # Test PairFormerBlock (minimal - pair bias only)
    print("\n7. PairFormerBlock (minimal):")
    block_min = PairFormerBlock(d_s, d_z, cond_dim, num_heads,
                                use_triangle_mult=False, use_triangle_attn=False)
    s_out, z_out = block_min(s, z, t_emb, mask)
    print(f"   s output shape: {s_out.shape}")
    print(f"   z output shape: {z_out.shape}")
    print("   PASSED")

    print("\nAll PairFormer component tests passed!")


if __name__ == "__main__":
    test_pairformer()
