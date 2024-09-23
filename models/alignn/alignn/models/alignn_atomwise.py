"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from typing import Tuple, Union
from torch.autograd import grad
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import AvgPooling
import torch

from typing import Literal
from torch import nn
from torch.nn import functional as F
from alignn.models.utils import RBFExpansion
from alignn.graphs import compute_bond_cosines
from alignn.utils import BaseSettings


class ALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_atomwise"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    grad_multiplier: int = -1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 0.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    force_mult_natoms: bool = False
    energy_mult_natoms: bool = False
    include_pos_deriv: bool = False
    use_cutoff_function: bool = False
    inner_cutoff: float = 6  # Ansgtrom
    stress_multiplier: float = 1
    add_reverse_forces: bool = False
    lg_on_fly: bool = False
    batch_stress: bool = True
    multiply_cutoff: bool = False
    extra_features: int = 0
    exponent: int = 3

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


def cutoff_function_based_edges(r, inner_cutoff=4, exponent=3):
    """Apply smooth cutoff to pairwise interactions."""
    ratio = r / inner_cutoff
    c1 = -(exponent + 1) * (exponent + 2) / 2
    c2 = exponent * (exponent + 2)
    c3 = -exponent * (exponent + 1) / 2
    envelope = (
        1
        + c1 * ratio**exponent
        + c2 * ratio ** (exponent + 1)
        + c3 * ratio ** (exponent + 2)
    )
    return torch.where(r <= inner_cutoff, envelope, torch.zeros_like(r))


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553."""

    def __init__(self, input_features: int, output_features: int, residual: bool = True):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        """Edge-gated graph convolution."""
        g = g.local_var()
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(self, in_features: int, out_features: int):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """Node and Edge updates for ALIGNN layer."""
        g = g.local_var()
        lg = lg.local_var()
        x, m = self.node_update(g, x, y)
        y, z = self.edge_update(lg, m, z)
        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNNAtomWise(nn.Module):
    """Atomistic Line graph network."""

    def __init__(self, config: ALIGNNAtomWiseConfig = ALIGNNAtomWiseConfig(name="alignn_atomwise")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.classification = config.classification
        self.config = config
        if self.config.gradwise_weight == 0:
            self.config.calculate_gradient = False

        self.atom_embedding = MLPLayer(config.atom_input_features, config.hidden_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=config.triplet_input_features),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [ALIGNNConv(config.hidden_features, config.hidden_features) for idx in range(config.alignn_layers)]
        )

        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(config.hidden_features, config.hidden_features) for idx in range(config.gcn_layers)]
        )

        self.readout = AvgPooling()

        if config.extra_features != 0:
            self.readout_feat = AvgPooling()
            self.extra_feature_embedding = MLPLayer(config.extra_features, config.extra_features)
            self.fc3 = nn.Linear(config.hidden_features + config.extra_features, config.output_features)
            self.fc1 = MLPLayer(config.extra_features + config.hidden_features, config.extra_features + config.hidden_features)
            self.fc2 = MLPLayer(config.extra_features + config.hidden_features, config.extra_features + config.hidden_features)

        if config.atomwise_output_features > 0:
            self.fc_atomwise = nn.Linear(config.hidden_features, config.atomwise_output_features)

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 1)
            self.softmax = nn.Sigmoid()
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]):
        """ALIGNN : start with `atom_features`."""
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()
            z = self.angle_embedding(lg.edata.pop("h"))

        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            features = self.extra_feature_embedding(features)

        g = g.local_var()
        result = {}

        # Initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        r = g.edata["r"]
        if self.config.calculate_gradient:
            r.requires_grad_(True)
        bondlength = torch.norm(r, dim=1)

        if self.config.lg_on_fly and len(self.alignn_layers) > 0:
            lg.ndata["r"] = r
            lg.apply_edges(compute_bond_cosines)
            z = self.angle_embedding(lg.edata.pop("h"))

        if self.config.use_cutoff_function:
            if self.config.multiply_cutoff:
                c_off = cutoff_function_based_edges(
                    bondlength, inner_cutoff=self.config.inner_cutoff, exponent=self.config.exponent
                ).unsqueeze(dim=1)
                y = self.edge_embedding(bondlength) * c_off
            else:
                bondlength = cutoff_function_based_edges(
                    bondlength, inner_cutoff=self.config.inner_cutoff, exponent=self.config.exponent
                )
                y = self.edge_embedding(bondlength)
        else:
            y = self.edge_embedding(bondlength)

        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        h = self.readout(g, x)
        result["embedding"] = h  # Store the pooled embeddings

        out = torch.empty(1)
        if self.config.output_features is not None:
            out = self.fc(h)
            if self.config.extra_features != 0:
                h_feat = self.readout_feat(g, features)
                h = torch.cat((h, h_feat), 1)
                h = self.fc1(h)
                h = self.fc2(h)
                out = self.fc3(h)
            else:
                out = torch.squeeze(out)

        atomwise_pred = torch.empty(1)
        if (
            self.config.atomwise_output_features > 0
            and self.config.atomwise_weight != 0
        ):
            atomwise_pred = self.fc_atomwise(x)

        forces = torch.empty(1)
        stress = torch.empty(1)

        if self.config.calculate_gradient:
            dx = r
            if self.config.energy_mult_natoms:
                en_out = out * g.num_nodes()
            else:
                en_out = out

            pair_forces = (
                self.config.grad_multiplier
                * grad(
                    en_out,
                    dx,
                    grad_outputs=torch.ones_like(en_out),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
            if self.config.force_mult_natoms:
                pair_forces *= g.num_nodes()

            g.edata["pair_forces"] = pair_forces
            g.update_all(fn.copy_e("pair_forces", "m"), fn.sum("m", "forces_ji"))
            forces = torch.squeeze(g.ndata["forces_ji"])

            if self.config.stresswise_weight != 0:
                stress = self.compute_stress(g, r, pair_forces)

        if self.link:
            out = self.link(out)

        if self.classification:
            out = self.softmax(out)

        result["out"] = out
        result["grad"] = forces
        result["stresses"] = stress
        result["atomwise_pred"] = atomwise_pred

        return result
