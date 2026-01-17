import time
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, TensorProduct


# -----------------------
# Debug helpers
# -----------------------
def _tstat(name, t, show_head: int = 0, show_pairs: bool = False):
    if t is None:
        print(f"[DBG] {name}: None")
        return
    if not torch.is_tensor(t):
        print(f"[DBG] {name}: type={type(t)}")
        return

    msg = f"[DBG] {name}: shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"

    if t.numel() == 0:
        print(msg + " (empty)")
        return

    if t.is_floating_point() or t.is_complex():
        finite = torch.isfinite(t)
        finite_ratio = finite.float().mean().item()
        msg += f" finite={finite_ratio:.6f}"
        if finite.any():
            tf = t[finite]
            msg += f" min={tf.min().item():.3e} max={tf.max().item():.3e} mean={tf.mean().item():.3e}"
        else:
            msg += " (no finite values)"
        print(msg)
    else:
        tmin = t.min().item()
        tmax = t.max().item()
        tmean = t.to(torch.float64).mean().item()
        msg += f" min={tmin} max={tmax} mean={tmean:.3f}"
        print(msg)

    if show_head > 0:
        flat = t.reshape(-1)
        k = min(show_head, flat.numel())
        print(f"[DBG] {name} head({k}): {flat[:k].detach().cpu().tolist()}")

    # edge_index 专用：展示前几条 (dst,src) 对
    if show_pairs and t.dim() == 2 and t.size(0) == 2:
        k = min(10, t.size(1))
        pairs = t[:, :k].T.detach().cpu().tolist()
        print(f"[DBG] {name} first {k} edges (dst,src): {pairs}")


def _assert_finite(name, t, debug=False):
    if t is None or (not torch.is_tensor(t)):
        return
    if not (t.is_floating_point() or t.is_complex()):
        return
    if not torch.isfinite(t).all():
        if debug:
            _tstat(name, t)
            bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:10]
            print(f"[DBG] {name}: first non-finite indices: {bad.tolist()}")
        raise RuntimeError(f"[NaN/Inf] tensor '{name}' contains NaN/Inf")


def _maxabs(t: torch.Tensor) -> float:
    if (t is None) or (not torch.is_tensor(t)) or (t.numel() == 0):
        return float("nan")
    if not (t.is_floating_point() or t.is_complex()):
        return float("nan")
    tf = t[torch.isfinite(t)]
    if tf.numel() == 0:
        return float("inf")
    return tf.abs().max().item()


def prod(x):
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
    normalization_coefficients = []
    for ins in instructions:
        ins_dict = {
            'uvw': (irrep_in1[ins[0]].mul * irrep_in2[ins[1]].mul),
            'uvu': irrep_in2[ins[1]].mul,
            'uvv': irrep_in1[ins[0]].mul,
            'uuw': irrep_in1[ins[0]].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': irrep_in1[ins[0]].mul * (irrep_in2[ins[1]].mul - 1) // 2,
        }
        alpha = irrep_mid[ins[2]].ir.dim
        x = sum([ins_dict[ins[3]] for ins in instructions])
        if x > 0.0:
            alpha /= x
        normalization_coefficients += [math.sqrt(alpha)]

    irrep_mid, p, _ = irrep_mid.sort()
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train, alpha)
        for (i_in1, i_in2, i_out, mode, train), alpha
        in zip(instructions, normalization_coefficients)
    ]
    return irrep_mid, instructions


def cutoff_function(x, cutoff):
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)
    return torch.where(x < cutoff, torch.exp(-x_ ** 2 / ((cutoff - x_) * (cutoff + x_))), zeros)


class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super().__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha

        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]

        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float32))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float32))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float32))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))
        self.reset_parameters()

        self.debug = False
        self._dbg_printed = False

    def reset_parameters(self):
        nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.dim() == 1:
            r = r.unsqueeze(-1)

        r = r.clamp_min(1e-8)
        alpha = F.softplus(self._alpha)
        x = -alpha * r

        y = (-torch.expm1(x)).clamp_min(1e-15)
        logy = torch.log(y)

        v = self.v.view(1, -1).to(device=r.device, dtype=r.dtype)
        n = self.n.view(1, -1).to(device=r.device, dtype=r.dtype)
        logc = self.logc.view(1, -1).to(device=r.device, dtype=r.dtype)

        logy = torch.where(v == 0, torch.zeros_like(logy), logy)

        if self.debug and (not self._dbg_printed):
            self._dbg_printed = True
            rr = r.detach()
            print(f"[DBG] RBF: r finite={torch.isfinite(rr).float().mean().item():.6f} "
                  f"min={rr.min().item():.3e} max={rr.max().item():.3e} mean={rr.mean().item():.3e}")
            yy = y.detach()
            print(f"[DBG] RBF: y min={yy.min().item():.3e} max={yy.max().item():.3e} mean={yy.mean().item():.3e}")
            lly = logy.detach()
            print(f"[DBG] RBF: logy min={lly.min().item():.3e} max={lly.max().item():.3e} mean={lly.mean().item():.3e}")

        xx = logc + n * x + v * logy
        rbf = cutoff_function(r, self.cutoff.to(dtype=r.dtype)) * torch.exp(xx)
        return rbf


class NormGate(torch.nn.Module):
    def __init__(self, irrep):
        super().__init__()
        self.irrep = irrep
        self.norm = o3.Norm(self.irrep)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irrep:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(self.irrep[1:], o3.Irreps(f"{num_mul_wo_0}x0e"))
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul),
            nn.SiLU(),
            nn.Linear(num_mul, num_mul)
        )

    def forward(self, x):
        norm_x = self.norm(x)[:, self.irrep.slices()[0].stop:]
        f0 = torch.cat([x[:, self.irrep.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(x[:, self.irrep.slices()[0].stop:], gates[:, self.irrep.slices()[0].stop:])
        x = torch.cat([gates[:, self.irrep.slices()[0]], gated], dim=-1)
        return x


class InnerProduct(torch.nn.Module):
    def __init__(self, irrep_in):
        super().__init__()
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        irrep_out = o3.Irreps([(mul, "0e") for mul, _ in self.irrep_in])
        instr = [(i, i, i, "uuu", False, 1 / ir.dim) for i, (mul, ir) in enumerate(self.irrep_in)]
        self.tp = o3.TensorProduct(self.irrep_in, self.irrep_in, irrep_out, instr, irrep_normalization="component")

    def forward(self, features_1, features_2):
        return self.tp(features_1, features_2)


class ConvLayer(torch.nn.Module):
    """
    Update: add degree normalization after scatter(sum) to reduce blow-up.
    """
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
        normalize_by_degree: bool = True,   # NEW
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.edge_wise = edge_wise
        self.normalize_by_degree = normalize_by_degree

        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden if isinstance(irrep_hidden, o3.Irreps) else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode='uvu'
        )

        self.tp_node = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=False,
            internal_weights=False,
        )

        self.fc_node = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node.weight_numel],
            self.nonlinear_layer
        )

        num_mul = 0
        for mul, _ in self.irrep_in_node:
            num_mul += mul

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

        self.irrep_linear_out, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node)
        self.linear_node = Linear(self.irrep_in_node, self.irrep_linear_out, internal_weights=True, shared_weights=True, biases=True)
        self.linear_node_pre = Linear(self.irrep_in_node, self.irrep_linear_out, internal_weights=True, shared_weights=True, biases=True)

        self.inner_product = InnerProduct(self.irrep_in_node)

    def forward(self, data, x):
        # convention: edge_index = [dst, src]
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]

        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product(pre_x[edge_dst], pre_x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat(
                [pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                 pre_x[edge_dst][:, self.irrep_in_node.slices()[0]], s0],
                dim=-1
            )
            x = self.norm_gate(x)
            x = self.linear_node(x)
        else:
            s0 = self.inner_product(x[edge_dst], x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat(
                [x[edge_dst][:, self.irrep_in_node.slices()[0]],
                 x[edge_dst][:, self.irrep_in_node.slices()[0]], s0],
                dim=-1
            )

        self_x = x

        edge_features = self.tp_node(
            x[edge_src], data.edge_sh,
            self.fc_node(data.edge_attr) * self.layer_l0(s0)
        )

        if self.edge_wise:
            out = edge_features
        else:
            out = scatter(edge_features, edge_dst, dim=0, dim_size=len(x), reduce="sum")

            # NEW: degree normalization (sqrt)
            if self.normalize_by_degree:
                ones = torch.ones((edge_dst.numel(), 1), device=edge_dst.device, dtype=out.dtype)
                deg = scatter(ones, edge_dst, dim=0, dim_size=len(x), reduce="sum").clamp_min(1.0)
                out = out / deg.sqrt()

            # if you prefer fixed avg normalization:
            # if self.avg_num_neighbors is not None:
            #     out = out / math.sqrt(float(self.avg_num_neighbors))

        if self.irrep_in_node == self.irrep_out:
            out = out + self_x

        out = self.linear_out(out)
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
        super().__init__()
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden if isinstance(irrep_hidden, o3.Irreps) else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.resnet = resnet and self.irrep_in_node == self.irrep_out

        self.conv = ConvLayer(
            irrep_in_node=self.irrep_in_node,
            irrep_hidden=self.irrep_hidden,
            sh_irrep=self.sh_irrep,
            irrep_out=self.irrep_out,
            edge_attr_dim=edge_attr_dim,
            node_attr_dim=node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear='ssp',
            use_norm_gate=use_norm_gate,
            edge_wise=edge_wise,
            normalize_by_degree=True,   # NEW: stabilize
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
        super().__init__()
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu'
        )

        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node_pair.weight_numel],
            self.nonlinear_layer
        )

        self.inner_product = InnerProduct(self.irrep_in_node)
        self.norm_gate_pre = NormGate(self.irrep_tp_out_node_pair)
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)

        self.linear_node_pair_inner = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)
        self.linear_node_pair_n = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)

        self.linear_node_pair = Linear(self.irrep_tp_out_node_pair, self.irrep_out, internal_weights=True, shared_weights=True, biases=True)

        self.resnet = (self.irrep_in_node == self.irrep_out) and resnet

        num_mul = 0
        for mul, _ in self.irrep_in_node:
            num_mul += mul

        self.fc = nn.Sequential(
            nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel)
        )

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data.full_edge_index  # [dst, src]

        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat(
            [node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
             node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0],
            dim=-1
        )

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(
            node_attr[src],
            node_attr[dst],
            self.fc_node_pair(data.full_edge_attr) * self.fc(s0)
        )

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
        super().__init__()
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)

        self.resnet = resnet
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu'
        )

        self.linear_node_1 = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)
        self.linear_node_2 = Linear(self.irrep_in_node, self.irrep_in_node, internal_weights=True, shared_weights=True, biases=True)

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

        self.linear_node_3 = Linear(self.irrep_tp_out_node, self.irrep_out, internal_weights=True, shared_weights=True, biases=True)

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)

        x = self.tp(xl, xr)  # quadratic-like; can amplify if x already big

        if self.resnet:
            x = x + old_x

        x = self.norm_gate(x)
        x = self.linear_node_3(x)

        if self.resnet and old_fii is not None:
            x = old_fii + x
        return x


class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super().__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        if self.num_path_weight > 0:
            self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))

    def forward(self, x_in, weights=None, bias_weights=None):
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
                for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
            ]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            x1 = x_in_s[ins[0]].reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).to(x_in.device).type(x1.type())

            if weights is None:
                weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                result = torch.einsum("wuv, ijk, bwk->buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else:
                weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                result = torch.einsum("bwuv, bwk->buvk", weight, x1)
                if ins[0] == 0 and bias_weights is not None:
                    bw = bias_weights[:, bias_weight_index:bias_weight_index + prod(ins[-1][1:])].reshape([-1] + ins[-1][1:])
                    bias_weight_index += prod(ins[-1][1:])
                    result = result + bw.unsqueeze(-1)
                result = torch.einsum("ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])

            result = result.reshape(batch_num, self.irrep_out_1[ins[1]].dim, self.irrep_out_2[ins[2]].dim)
            key = (ins[1], ins[2])
            outputs[key] = outputs.get(key, 0) + result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                blocks.append(outputs.get(
                    (i, j),
                    torch.zeros((batch_num, self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                                device=x_in.device, dtype=x_in.dtype)
                ))
            rows.append(torch.cat(blocks, dim=-1))
        return torch.cat(rows, dim=-2)

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions


class QHNet(nn.Module):
    def __init__(self,
                 in_node_features=1,
                 sh_lmax=4,
                 hidden_size=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=12,
                 num_nodes=20,
                 radius_embed_dim=32,
                 convention="thu_cluster"):
        super().__init__()
        self.convention = convention
        self.order = sh_lmax

        # DEBUG 开关：你外部设 self.model.debug=True 就会逐层打印
        self.debug = False

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
        self.hidden_bottle_irrep_base = o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e')

        self.input_irrep = o3.Irreps(f'{self.hs}x0e')

        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)

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
                    invariant_layers=1,
                    invariant_neurons=self.hs,
                    resnet=True,
                ))

        if convention == 'pyscf_6311_plus_gdp':
            out_irreps = "5x0e + 4x1e + 1x2e"
        elif convention == 'thu_cluster':
            out_irreps = "4x0e + 3x1e + 1x2e"
        else:
            out_irreps = "3x0e + 2x1e + 1x2e"

        self.expand_ii = nn.ModuleDict()
        self.expand_ij = nn.ModuleDict()
        self.fc_ii = nn.ModuleDict()
        self.fc_ij = nn.ModuleDict()
        self.fc_ii_bias = nn.ModuleDict()
        self.fc_ij_bias = nn.ModuleDict()

        for name in {"hamiltonian"}:
            input_expand = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

            self.expand_ii[name] = Expansion(input_expand, o3.Irreps(out_irreps), o3.Irreps(out_irreps))
            self.fc_ii[name] = nn.Sequential(nn.Linear(self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ii[name].num_path_weight))
            self.fc_ii_bias[name] = nn.Sequential(nn.Linear(self.hs, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ii[name].num_bias))

            self.expand_ij[name] = Expansion(input_expand, o3.Irreps(out_irreps), o3.Irreps(out_irreps))
            self.fc_ij[name] = nn.Sequential(nn.Linear(self.hs * 2, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ij[name].num_path_weight))
            self.fc_ij_bias[name] = nn.Sequential(nn.Linear(self.hs * 2, self.hs), nn.SiLU(), nn.Linear(self.hs, self.expand_ij[name].num_bias))

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    @property
    def device(self):
        return next(self.parameters()).device

    def _edge_features(self, pos: torch.Tensor, edge_index: torch.Tensor, tag: str = ""):
        dst = edge_index[0].long()
        src = edge_index[1].long()

        if self.debug:
            _assert_finite(f"{tag}pos", pos, debug=True)
            N = pos.size(0)
            print(f"[DBG] {tag}edge_index ranges: dst[{int(dst.min())},{int(dst.max())}] src[{int(src.min())},{int(src.max())}] N={N} E={dst.numel()}")
            if int(dst.min()) < 0 or int(src.min()) < 0 or int(dst.max()) >= N or int(src.max()) >= N:
                _tstat(f"{tag}edge_index", edge_index, show_pairs=True)
                raise RuntimeError(f"[OOB] {tag}edge_index out of bounds")

        edge_vec = pos[dst] - pos[src]
        r = edge_vec.norm(dim=-1, keepdim=True)

        if self.debug:
            _tstat(f"{tag}r", r)
            print(f"[DBG] {tag}r max={r.max().item():.6e} min={r.min().item():.6e}")

        self.distance_expansion.debug = bool(self.debug)
        rbf = self.distance_expansion(r).to(dtype=pos.dtype)

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component'
        ).to(dtype=pos.dtype)

        if self.debug:
            _assert_finite(f"{tag}rbf", rbf, debug=True)
            _assert_finite(f"{tag}edge_sh", edge_sh, debug=True)

        return rbf, edge_sh

    def build_graph(self, data, max_radius):
        node_attr = data.atoms.view(-1)

        edge_index = radius_graph(data.pos, max_radius, data.batch, max_num_neighbors=data.num_nodes)

        if self.debug:
            _tstat("local/edge_index", edge_index, show_pairs=True)
            # degree 分布
            dst = edge_index[0]
            ones = torch.ones((dst.numel(), 1), device=dst.device, dtype=torch.float32)
            deg = scatter(ones, dst, dim=0, dim_size=int(data.pos.size(0)), reduce="sum").squeeze(-1)
            print(f"[DBG] local/deg: min={deg.min().item():.1f} max={deg.max().item():.1f} mean={deg.mean().item():.2f}")

        edge_attr, edge_sh = self._edge_features(data.pos, edge_index, tag="local/")
        return node_attr, edge_index, edge_attr, edge_sh

    def _transpose_index_by_keys(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        dst = edge_index[0].long()
        src = edge_index[1].long()
        key = src * num_nodes + dst
        rev_key = dst * num_nodes + src
        skey, perm = torch.sort(key)
        pos = torch.searchsorted(skey, rev_key)
        if not torch.all(pos < skey.numel()):
            raise RuntimeError("transpose_index: reverse edge missing (searchsorted out of range)")
        if not torch.all(skey[pos] == rev_key):
            raise RuntimeError("transpose_index: reverse edge missing (key mismatch)")
        return perm[pos]

    def build_full_graph_from_data(self, data):
        if hasattr(data, "edge_index_full") and data.edge_index_full is not None:
            full_edge_index = data.edge_index_full
        else:
            full_edge_index = radius_graph(data.pos, 1e9, data.batch, max_num_neighbors=data.num_nodes)

        if self.debug:
            _tstat("full/edge_index", full_edge_index, show_pairs=True)
            dst = full_edge_index[0]
            ones = torch.ones((dst.numel(), 1), device=dst.device, dtype=torch.float32)
            deg = scatter(ones, dst, dim=0, dim_size=int(data.pos.size(0)), reduce="sum").squeeze(-1)
            print(f"[DBG] full/deg: min={deg.min().item():.1f} max={deg.max().item():.1f} mean={deg.mean().item():.2f}")

        full_edge_attr, full_edge_sh = self._edge_features(data.pos, full_edge_index, tag="full/")

        N = int(data.pos.shape[0])
        transpose_edge_index = self._transpose_index_by_keys(full_edge_index, num_nodes=N)
        return full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index

    def forward(self, data, keep_blocks=True):
        if self.debug:
            _tstat("pos(in)", data.pos)
            _tstat("atoms(in)", data.atoms)
            _assert_finite("pos(in)", data.pos, debug=True)

        node_attr, edge_index, edge_attr, edge_sh = self.build_graph(data, self.max_radius)

        # embedding index 检查
        if self.debug:
            max_z = int(node_attr.max().item())
            if max_z >= self.node_embedding.num_embeddings:
                raise RuntimeError(f"[OOB] atom Z max={max_z} >= embedding size={self.node_embedding.num_embeddings}")

        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = node_attr, edge_index, edge_attr, edge_sh

        full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = self.build_full_graph_from_data(data)
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = full_edge_index, full_edge_attr, full_edge_sh

        if self.debug:
            _tstat("edge_attr(local)", data.edge_attr)
            _tstat("full_edge_attr", data.full_edge_attr)
            _tstat("edge_index_full(in)", getattr(data, "edge_index_full", None), show_pairs=True)
            _tstat("full_edge_index(used)", data.full_edge_index, show_pairs=True)

        # ---- layer-by-layer debug ----
        fii = None
        fij = None

        if self.debug:
            print(f"[DBG] node_attr(embed) maxabs={_maxabs(node_attr):.3e}")
            _assert_finite("node_attr(embed)", node_attr, debug=True)

        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(data, node_attr)

            if self.debug:
                print(f"[DBG] node_attr@conv{layer_idx} maxabs={_maxabs(node_attr):.3e}")
                _assert_finite(f"node_attr@conv{layer_idx}", node_attr, debug=True)

            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx - self.start_layer - 1](data, node_attr, fii)
                fij = self.e3_gnn_node_pair_layer[layer_idx - self.start_layer - 1](data, node_attr, fij)

                if self.debug:
                    print(f"[DBG] fii@self{layer_idx} maxabs={_maxabs(fii):.3e}")
                    _assert_finite(f"fii@self{layer_idx}", fii, debug=True)
                    print(f"[DBG] fij@pair{layer_idx} maxabs={_maxabs(fij):.3e}")
                    _assert_finite(f"fij@pair{layer_idx}", fij, debug=True)

        if fii is None or fij is None:
            raise RuntimeError(
                f"fii/fij is None. num_gnn_layers={self.num_gnn_layers} must be > start_layer={self.start_layer}."
            )

        fii = self.output_ii(fii)
        fij = self.output_ij(fij)

        if self.debug:
            print(f"[DBG] fii(after output_ii) maxabs={_maxabs(fii):.3e}")
            _assert_finite("fii", fii, debug=True)
            print(f"[DBG] fij(after output_ij) maxabs={_maxabs(fij):.3e}")
            _assert_finite("fij", fij, debug=True)

        ham_ii = self.expand_ii['hamiltonian'](
            fii,
            self.fc_ii['hamiltonian'](data.node_attr),
            self.fc_ii_bias['hamiltonian'](data.node_attr)
        )

        full_dst, full_src = data.full_edge_index
        node_pair_embedding = torch.cat([data.node_attr[full_dst], data.node_attr[full_src]], dim=-1)
        ham_ij = self.expand_ij['hamiltonian'](
            fij,
            self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding)
        )

        if self.debug:
            _assert_finite("ham_ii(raw)", ham_ii, debug=True)
            _assert_finite("ham_ij(raw)", ham_ij, debug=True)

        if keep_blocks is False:
            raise NotImplementedError("keep_blocks=False not maintained in this debug build.")
        else:
            ret_ham_ii = ham_ii + ham_ii.transpose(-1, -2)
            ret_ham_ij = ham_ij + ham_ij[transpose_edge_index].transpose(-1, -2)
            return {
                'hamiltonian_diagonal_blocks': ret_ham_ii,
                'hamiltonian_non_diagonal_blocks': ret_ham_ij
            }