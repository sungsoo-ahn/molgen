"""Flow matching implementation for molecular graph generation.

Implements Conditional Flow Matching (CFM) with linear interpolation paths
for training, and ODE-based sampling for generation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.dit import GraphDiT


class FlowMatchingScheduler:
    """Scheduler for flow matching training.

    Implements linear interpolation paths from noise (t=0) to data (t=1):
        x_t = (1 - t) * x_0 + t * x_1

    The target velocity is simply:
        v = x_1 - x_0

    Args:
        sigma_min: Minimum noise scale for stability (added to prior samples)
    """

    def __init__(self, sigma_min: float = 0.001):
        self.sigma_min = sigma_min

    def get_train_sample(
        self,
        x_1: torch.Tensor,
        adj_1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample noisy state and compute target velocity for training.

        Args:
            x_1: Data node features [batch, N, A]
            adj_1: Data adjacency [batch, N, N, E]
            t: Timesteps [batch] in [0, 1]

        Returns:
            x_t: Interpolated node features [batch, N, A]
            adj_t: Interpolated adjacency [batch, N, N, E]
            v_x_target: Target node velocity [batch, N, A]
            v_adj_target: Target edge velocity [batch, N, N, E]
        """
        # Sample from prior (standard Gaussian)
        x_0 = torch.randn_like(x_1)
        adj_0 = torch.randn_like(adj_1)

        # Expand timestep for broadcasting
        t_x = t.view(-1, 1, 1)  # [B, 1, 1] for nodes
        t_adj = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1] for edges

        # Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - t_x) * x_0 + t_x * x_1
        adj_t = (1 - t_adj) * adj_0 + t_adj * adj_1

        # Target velocity is just the direction from noise to data
        v_x_target = x_1 - x_0
        v_adj_target = adj_1 - adj_0

        return x_t, adj_t, v_x_target, v_adj_target

    def sample_prior(
        self,
        batch_size: int,
        max_atoms: int,
        num_atom_types: int,
        num_bond_types: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from prior distribution (Gaussian).

        Args:
            batch_size: Number of samples
            max_atoms: Maximum atoms per molecule
            num_atom_types: Number of atom types
            num_bond_types: Number of bond types
            device: Device for tensors

        Returns:
            x_0: Prior node features [batch, max_atoms, num_atom_types]
            adj_0: Prior adjacency [batch, max_atoms, max_atoms, num_bond_types]
        """
        x_0 = torch.randn(batch_size, max_atoms, num_atom_types, device=device)
        adj_0 = torch.randn(
            batch_size, max_atoms, max_atoms, num_bond_types, device=device
        )

        # Add small noise for stability
        x_0 = x_0 * (1.0 + self.sigma_min)
        adj_0 = adj_0 * (1.0 + self.sigma_min)

        return x_0, adj_0


class FlowMatchingSampler:
    """ODE-based sampler for flow matching models.

    Integrates the learned velocity field from t=0 (noise) to t=1 (data)
    using numerical ODE solvers.

    Args:
        model: Trained GraphDiT model
        scheduler: FlowMatchingScheduler instance
        num_steps: Number of integration steps
        method: ODE solver method ('euler', 'heun', 'rk4')
    """

    def __init__(
        self,
        model: GraphDiT,
        scheduler: FlowMatchingScheduler,
        num_steps: int = 100,
        method: str = "heun",
    ):
        self.model = model
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.method = method

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate molecular graphs via ODE integration.

        Integrates from t=0 (noise) to t=1 (data) using the learned velocity field.

        Args:
            batch_size: Number of molecules to generate
            mask: Node mask [batch_size, max_atoms] defining molecule sizes
            device: Device for computation

        Returns:
            x: Final node features [batch, max_atoms, num_atom_types]
            adj: Final adjacency [batch, max_atoms, max_atoms, num_bond_types]
        """
        self.model.eval()

        # Get dimensions from model
        max_atoms = self.model.max_atoms
        num_atom_types = self.model.num_atom_types
        num_bond_types = self.model.num_bond_types

        # Initialize from prior
        x_t, adj_t = self.scheduler.sample_prior(
            batch_size, max_atoms, num_atom_types, num_bond_types, device
        )

        # Time stepping
        dt = 1.0 / self.num_steps
        t_steps = torch.linspace(0, 1 - dt, self.num_steps, device=device)

        for t in t_steps:
            t_batch = torch.full((batch_size,), t.item(), device=device)

            if self.method == "euler":
                x_t, adj_t = self._euler_step(x_t, adj_t, t_batch, mask, dt)
            elif self.method == "heun":
                x_t, adj_t = self._heun_step(x_t, adj_t, t_batch, mask, dt)
            elif self.method == "rk4":
                x_t, adj_t = self._rk4_step(x_t, adj_t, t_batch, mask, dt)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        return x_t, adj_t

    def _euler_step(
        self,
        x_t: torch.Tensor,
        adj_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler step: x_{t+dt} = x_t + dt * v(x_t, t)."""
        v_x, v_adj = self.model(x_t, adj_t, t, mask)
        x_next = x_t + dt * v_x
        adj_next = adj_t + dt * v_adj
        return x_next, adj_next

    def _heun_step(
        self,
        x_t: torch.Tensor,
        adj_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Heun's method (2nd order): average of slopes at start and end."""
        # First evaluation
        v_x_1, v_adj_1 = self.model(x_t, adj_t, t, mask)

        # Euler prediction
        x_pred = x_t + dt * v_x_1
        adj_pred = adj_t + dt * v_adj_1

        # Second evaluation at predicted point
        t_next = t + dt
        v_x_2, v_adj_2 = self.model(x_pred, adj_pred, t_next, mask)

        # Average velocities
        x_next = x_t + 0.5 * dt * (v_x_1 + v_x_2)
        adj_next = adj_t + 0.5 * dt * (v_adj_1 + v_adj_2)

        return x_next, adj_next

    def _rk4_step(
        self,
        x_t: torch.Tensor,
        adj_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """4th-order Runge-Kutta step."""
        half_dt = dt / 2

        # k1
        k1_x, k1_adj = self.model(x_t, adj_t, t, mask)

        # k2
        t_half = t + half_dt
        k2_x, k2_adj = self.model(
            x_t + half_dt * k1_x,
            adj_t + half_dt * k1_adj,
            t_half,
            mask,
        )

        # k3
        k3_x, k3_adj = self.model(
            x_t + half_dt * k2_x,
            adj_t + half_dt * k2_adj,
            t_half,
            mask,
        )

        # k4
        t_next = t + dt
        k4_x, k4_adj = self.model(
            x_t + dt * k3_x,
            adj_t + dt * k3_adj,
            t_next,
            mask,
        )

        # Combine
        x_next = x_t + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        adj_next = adj_t + (dt / 6) * (k1_adj + 2 * k2_adj + 2 * k3_adj + k4_adj)

        return x_next, adj_next


def flow_matching_loss(
    model: GraphDiT,
    x_1: torch.Tensor,
    adj_1: torch.Tensor,
    mask: torch.Tensor,
    scheduler: FlowMatchingScheduler,
) -> Tuple[torch.Tensor, dict]:
    """Compute flow matching training loss.

    Loss is MSE between predicted and target velocity, masked to only
    consider valid atoms and edges.

    Args:
        model: GraphDiT model
        x_1: Data node features [batch, N, A]
        adj_1: Data adjacency [batch, N, N, E]
        mask: Node mask [batch, N]
        scheduler: FlowMatchingScheduler instance

    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary with component losses
    """
    B = x_1.shape[0]
    device = x_1.device

    # Sample uniform timesteps
    t = torch.rand(B, device=device)

    # Get noisy samples and targets
    x_t, adj_t, v_x_target, v_adj_target = scheduler.get_train_sample(x_1, adj_1, t)

    # Predict velocity
    v_x_pred, v_adj_pred = model(x_t, adj_t, t, mask)

    # Compute masked MSE for nodes
    # Mask: [B, N, 1] for broadcasting
    node_mask = mask.unsqueeze(-1).float()  # [B, N, 1]
    node_diff_sq = (v_x_pred - v_x_target) ** 2  # [B, N, A]
    loss_nodes = (node_diff_sq * node_mask).sum() / (node_mask.sum() * v_x_pred.shape[-1])

    # Compute masked MSE for edges
    # Edge mask: valid if both endpoints are valid
    edge_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1).float()  # [B, N, N, 1]
    edge_diff_sq = (v_adj_pred - v_adj_target) ** 2  # [B, N, N, E]
    loss_edges = (edge_diff_sq * edge_mask).sum() / (edge_mask.sum() * v_adj_pred.shape[-1])

    # Total loss
    loss = loss_nodes + loss_edges

    metrics = {
        "loss_nodes": loss_nodes.item(),
        "loss_edges": loss_edges.item(),
        "loss_total": loss.item(),
    }

    return loss, metrics


class FlowMatchingModule(nn.Module):
    """Wrapper module combining model and scheduler for easy checkpoint saving.

    This module wraps GraphDiT and exposes training/sampling interfaces.

    Args:
        model: GraphDiT instance
        scheduler: FlowMatchingScheduler instance
    """

    def __init__(
        self,
        model: GraphDiT,
        scheduler: Optional[FlowMatchingScheduler] = None,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler or FlowMatchingScheduler()

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x, adj, t, mask)

    def compute_loss(
        self,
        x_1: torch.Tensor,
        adj_1: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute flow matching loss."""
        return flow_matching_loss(self.model, x_1, adj_1, mask, self.scheduler)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        mask: torch.Tensor,
        device: torch.device,
        num_steps: int = 100,
        method: str = "heun",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples using ODE integration."""
        sampler = FlowMatchingSampler(
            self.model, self.scheduler, num_steps=num_steps, method=method
        )
        return sampler.sample(batch_size, mask, device)
