import torch.nn.functional as F
from .node_wise import ExponentialBernsteinRadialBasisFunctions
from .utils import get_nonlinear

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet, Gate, Activation
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from .Expanson import Expansion


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


def get_nonlinear(nonlinear: str):
    if nonlinear.lower() == 'ssp':
        return ShiftedSoftPlus
    elif nonlinear.lower() == 'silu':
        return F.silu
    elif nonlinear.lower() == 'tanh':
        return F.tanh
    elif nonlinear.lower() == 'abs':
        return torch.abs
    else:
        raise NotImplementedError


def get_feasible_irrep(irrep_in1, irrep_in2, cutoff_irrep_out, tp_mode="uvu"):
    irrep_mid = []
    instructions = []

    for i, (_, ir_in) in enumerate(irrep_in1):
        for j, (_, ir_edge) in enumerate(irrep_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in cutoff_irrep_out:
                    if (cutoff_irrep_out.count(ir_out), ir_out) not in irrep_mid:
                        k = len(irrep_mid)
                        irrep_mid.append((cutoff_irrep_out.count(ir_out), ir_out))
                    else:
                        k = irrep_mid.index((cutoff_irrep_out.count(ir_out), ir_out))
                    instructions.append((i, j, k, tp_mode, True))

    irrep_mid = o3.Irreps(irrep_mid)
    irrep_mid, p, _ = irrep_mid.sort()
    instructions = [(i_in1, i_in2, p[i_out], mode, train)
        for (i_in1, i_in2, i_out, mode, train) in instructions]
    return irrep_mid, instructions


class NormGate(torch.nn.Module):
    def __init__(self, irrep):
        super(NormGate, self).__init__()
        self.irrep = irrep
        self.norm = o3.Norm(self.irrep)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irrep:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(
            self.irrep[1:], o3.Irreps(f"{num_mul_wo_0}x0e"))
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul),
            nn.SiLU(),
            nn.Linear(num_mul, num_mul))

        self.num_mul = num_mul
        self.num_mul_wo_0 = num_mul_wo_0

    def forward(self, x):
        norm_x = self.norm(x)[:, self.irrep.slices()[0].stop:]
        f0 = torch.cat([x[:, self.irrep.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(x[:, self.irrep.slices()[0].stop:], gates[:, self.irrep.slices()[0].stop:])
        x = torch.cat([gates[:, self.irrep.slices()[0]], gated], dim=-1)
        return x


class ConvLayer(torch.nn.Module):
    def __init__(
            self,
            irrep_in_node,
            irrep_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear='ssp',
            use_norm_gate=True,
            edge_wise=False,
    ):
        super(ConvLayer, self).__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.edge_wise = edge_wise

        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden \
            if isinstance(irrep_hidden, o3.Irreps) else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode='uvu')

        self.tp_node = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=False,
            internal_weights=False,
            irrep_normalization='none'
        )

        self.fc_node = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node.weight_numel],
            self.nonlinear_layer
        )

        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.layer_l0 = FullyConnectedNet(
            [num_mul + self.irrep_in_node[0][0]] + invariant_layers * [invariant_neurons] + [self.tp_node.weight_numel],
            self.nonlinear_layer
        )

        self.linear_out = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.use_norm_gate = use_norm_gate
        self.norm_gate = NormGate(self.irrep_in_node)
        self.irrep_linear_out, instruction_node = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node)
        self.linear_node = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.linear_node_pre = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.inner_product = InnerProduct(self.irrep_in_node)

    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]

        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product(pre_x[edge_dst], pre_x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                            pre_x[edge_dst][:, self.irrep_in_node.slices()[0]], s0], dim=-1)
            x = self.norm_gate(x)
            x = self.linear_node(x)
        else:
            s0 = self.inner_product(x[edge_dst], x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([x[edge_dst][:, self.irrep_in_node.slices()[0]],
                            x[edge_dst][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        self_x = x

        edge_features = self.tp_node(
            x[edge_src], data.edge_sh, self.fc_node(data.edge_attr) * self.layer_l0(s0))

        if self.edge_wise:
            out = edge_features
        else:
            out = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        if self.irrep_in_node == self.irrep_out:
            out = out + self_x

        out = self.linear_out(out)
        return out


class InnerProduct(torch.nn.Module):
    def __init__(self, irrep_in):
        super(InnerProduct, self).__init__()
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        irrep_out = o3.Irreps([(mul, "0e") for mul, _ in self.irrep_in])
        instr = [(i, i, i, "uuu", False, 1/ir.dim) for i, (mul, ir) in enumerate(self.irrep_in)]
        self.tp = o3.TensorProduct(self.irrep_in, self.irrep_in, irrep_out, instr, irrep_normalization="component")
        self.irrep_out = irrep_out.simplify()

    def forward(self, features_1, features_2):
        out = self.tp(features_1, features_2)
        return out


class ConvNetLayer(torch.nn.Module):
    def __init__(
            self,
            irrep_in_node,
            irrep_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            resnet: bool = True,
            use_norm_gate=True,
            edge_wise=False,
    ):
        super(ConvNetLayer, self).__init__()
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden if isinstance(irrep_hidden, o3.Irreps) \
            else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet and self.irrep_in_node == self.irrep_out

        self.conv = ConvLayer(
            irrep_in_node=self.irrep_in_node,
            irrep_hidden=self.irrep_hidden,
            sh_irrep=self.sh_irrep,
            irrep_out=self.irrep_out,
            edge_attr_dim=self.edge_attr_dim,
            node_attr_dim=self.node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear='ssp',
            use_norm_gate=use_norm_gate,
            edge_wise=edge_wise
        )

    def forward(self, data, x):
        old_x = x
        x = self.conv(data, x)
        if self.resnet and self.irrep_out == self.irrep_in_node:
            x = old_x + x
        return x


class PairNetLayer(torch.nn.Module):
    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 invariant_layers=1,
                 invariant_neurons=8,
                 nonlinear='ssp'):
        super(PairNetLayer, self).__init__()
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden \
            if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu')

        self.irrep_tp_out_node_pair_msg, instruction_node_pair_msg = get_feasible_irrep(
            self.irrep_tp_in_node, self.sh_irrep, self.irrep_bottle_hidden, tp_mode='uvu')

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_pair_n = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.irrep_tp_out_node_pair_2, instruction_node_pair_2 = get_feasible_irrep(
            self.irrep_tp_out_node_pair, self.irrep_tp_out_node_pair, self.irrep_bottle_hidden, tp_mode='uuu')

        self.tp_node_pair_2 = TensorProduct(
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair_2,
            instruction_node_pair_2,
            shared_weights=True,
            internal_weights=True
        )


        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node_pair.weight_numel],
            self.nonlinear_layer
        )

        self.linear_node_pair_2 = Linear(
            irreps_in=self.irrep_tp_out_node_pair_2,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        self.inner_product = InnerProduct(self.irrep_in_node)
        self.norm = o3.Norm(self.irrep_in_node)
        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.norm_gate_pre = NormGate(self.irrep_tp_out_node_pair)
        self.fc = nn.Sequential(
            nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data.full_edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat([node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
                        node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],
            self.fc_node_pair(data.full_edge_attr) * self.fc(s0))

        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair


class SelfNetLayer(torch.nn.Module):
    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 nonlinear='ssp'):
        super(SelfNetLayer, self).__init__()
        self.sh_irrep = sh_irrep
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden \
            if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu')

        # - Build modules -
        self.linear_node_1 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_2 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.tp = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=True,
            internal_weights=True
        )
        self.norm_gate = NormGate(self.irrep_out)
        self.norm_gate_1 = NormGate(self.irrep_in_node)
        self.norm_gate_2 = NormGate(self.irrep_in_node)
        self.linear_node_3 = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)
        x = self.tp(xl, xr)
        if self.resnet:
            x = x + old_x
        x = self.norm_gate(x)
        x = self.linear_node_3(x)
        if self.resnet and old_fii is not None:
            x = old_fii + x
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class QHNet(nn.Module):
    def __init__(self,
                 in_node_features=1,
                 sh_lmax=4,
                 hidden_size=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=12,
                 num_nodes=10,
                 radius_embed_dim=32):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super(QHNet, self).__init__()
        # store hyperparameter values
        self.atom_orbs = [
            [[8, 0, '1s'], [8, 0, '2s'], [8, 0, '3s'], [8, 1, '2p'], [8, 1, '3p'], [8, 2, '3d']],
            [[1, 0, '1s'], [1, 0, '2s'], [1, 1, '2p']],
            [[1, 0, '1s'], [1, 0, '2s'], [1, 1, '2p']]
        ]
        self.order = sh_lmax

        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = hidden_size
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        self.hidden_irrep = o3.Irreps(f'{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e')
        self.hidden_bottle_irrep = o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1o + {self.hbs}x2e + {self.hbs}x3o + {self.hbs}x4e')
        self.hidden_irrep_base = o3.Irreps(f'{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e')
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.num_fc_layer = 1

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.start_layer = 2
        for i in range(self.num_gnn_layers):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.e3_gnn_layer.append(ConvNetLayer(
                irrep_in_node=input_irrep,
                irrep_hidden=self.hidden_irrep,
                irrep_out=self.hidden_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                sh_irrep=self.sh_irrep,
                resnet=True,
                use_norm_gate=True if i != 0 else False
            ))

            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                ))
        self.nonlinear_layer = get_nonlinear('ssp')

        input_expand_ii = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")
        self.expand_ii = Expansion(
            input_expand_ii, o3.Irreps("3x0e + 2x1e + 1x2e"), o3.Irreps("3x0e + 2x1e + 1x2e"))
        self.fc_ii = torch.nn.Sequential(
            nn.Linear(self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ii.num_path_weight))
        self.fc_ii_bias = torch.nn.Sequential(
            nn.Linear(self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ii.num_bias))

        self.expand_ij = Expansion(
            input_expand_ii, o3.Irreps("3x0e + 2x1e + 1x2e"), o3.Irreps("3x0e + 2x1e + 1x2e"))
        self.fc_ij = torch.nn.Sequential(
            nn.Linear(self.hs * 2, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ij.num_path_weight))
        self.fc_ij_bias = torch.nn.Sequential(
            nn.Linear(self.hs * 2, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ij.num_bias))

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    def set(self, device):
        self = self.to(device)
        self.orbital_mask = self.get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, keep_blocks=True):
        node_attr, edge_index, rbf_new, edge_sh, _ = self.build_graph(data, self.max_radius)
        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = \
            node_attr, edge_index, rbf_new, edge_sh

        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = self.build_graph(data, 10000)
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = \
            full_edge_index, full_edge_attr, full_edge_sh

        full_dst, full_src = data.full_edge_index

        fii = None
        fij = None
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(data, node_attr)
            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](data, node_attr, fii)
                fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](data, node_attr, fij)

        fii = self.output_ii(fii)
        fij = self.output_ij(fij)
        hamiltonian_diagonal_matrix = self.expand_ii(
            fii, self.fc_ii(data.node_attr), self.fc_ii_bias(data.node_attr))
        node_pair_embedding = torch.cat([data.node_attr[full_dst], data.node_attr[full_src]], dim=-1)
        hamiltonian_non_diagonal_matrix = self.expand_ij(
            fij, self.fc_ij(node_pair_embedding), self.fc_ij_bias(node_pair_embedding))

        if keep_blocks is False:
            hamiltonian_matrix = self.build_final_matrix(
                data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix)
            hamiltonian_matrix = hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)
            results = {}
            results['hamiltonian'] = hamiltonian_matrix

        else:
            ret_hamiltonian_diagonal_matrix = \
                hamiltonian_diagonal_matrix + hamiltonian_diagonal_matrix.transpose(-1, -2)
            ret_hamiltonian_non_diagonal_matrix = hamiltonian_non_diagonal_matrix + \
                      hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)

            results = {}
            results['hamiltonian_diagonal_blocks'] = ret_hamiltonian_diagonal_matrix
            results['hamiltonian_non_diagonal_blocks'] = ret_hamiltonian_non_diagonal_matrix

        return results

    def build_graph(self, data, max_radius):
        node_attr = data.atoms.squeeze()
        
        
        radius_edges = radius_graph(data.pos, max_radius, data.batch, max_num_neighbors=data.num_nodes)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.distance_expansion(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        final_matrix = []
        dst, src = data.full_edge_index
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                matrix_col = []
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                    if src_idx == dst_idx:
                        matrix_col.append(diagonal_matrix[src_idx].index_select(
                            -2, self.orbital_mask[data.atoms[dst_idx].item()]).index_select(
                            -1, self.orbital_mask[data.atoms[src_idx].item()])
                        )
                    else:
                        mask1 = (src == src_idx)
                        mask2 = (dst == dst_idx)
                        index = torch.where(mask1 & mask2)[0].item()

                        matrix_col.append(
                            non_diagonal_matrix[index].index_select(
                                -2, self.orbital_mask[data.atoms[dst_idx].item()]).index_select(
                                -1, self.orbital_mask[data.atoms[src_idx].item()]))
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))
        final_matrix = torch.stack(final_matrix, dim=0)
        return final_matrix

    def get_orbital_mask(self):
        idx_1s_2s = torch.tensor([0, 1])
        idx_2p = torch.tensor([3, 4, 5])
        orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
        orbital_mask_line2 = torch.arange(14)
        orbital_mask = {}
        for i in range(1, 11):
            orbital_mask[i] = orbital_mask_line1 if i <=2 else orbital_mask_line2
        return orbital_mask

    def split_matrix(self, data):
        diagonal_matrix, non_diagonal_matrix = \
            torch.zeros(data.atoms.shape[0], 14, 14).type(data.pos.type()).to(self.device), \
            torch.zeros(data.edge_index.shape[1], 14, 14).type(data.pos.type()).to(self.device)

        data.matrix =  data.matrix.reshape(
            len(data.ptr) - 1, data.matrix.shape[-1], data.matrix.shape[-1])

        num_atoms = 0
        num_edges = 0
        for graph_idx in range(data.ptr.shape[0] - 1):
            slices = [0]
            for atom_idx in data.atoms[range(data.ptr[graph_idx], data.ptr[graph_idx + 1])]:
                slices.append(slices[-1] + len(self.orbital_mask[atom_idx.item()]))

            for node_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                node_idx = node_idx - num_atoms
                orb_mask = self.orbital_mask[data.atoms[node_idx].item()]
                diagonal_matrix[node_idx][orb_mask][:, orb_mask] = \
                    data.matrix[graph_idx][slices[node_idx]: slices[node_idx+1], slices[node_idx]: slices[node_idx+1]]

            for edge_index_idx in range(num_edges, data.edge_index.shape[1]):
                dst, src = data.edge_index[:, edge_index_idx]
                if dst > data.ptr[graph_idx + 1] or src > data.ptr[graph_idx + 1]:
                    break
                num_edges = num_edges + 1
                orb_mask_dst = self.orbital_mask[data.atoms[dst].item()]
                orb_mask_src = self.orbital_mask[data.atoms[src].item()]
                graph_dst, graph_src = dst - num_atoms, src - num_atoms
                non_diagonal_matrix[edge_index_idx][orb_mask_dst][:, orb_mask_src] = \
                    data.matrix[graph_idx][slices[graph_dst]: slices[graph_dst+1], slices[graph_src]: slices[graph_src+1]]

            num_atoms = num_atoms + data.ptr[graph_idx + 1] - data.ptr[graph_idx]
        return diagonal_matrix, non_diagonal_matrix
