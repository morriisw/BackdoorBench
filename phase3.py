"""
Phase 3: learned concept removal.

Phases 0-2 fix the direction (a difference of means) and the scope (a
hand-chosen cosine threshold). Phase 3 replaces them one at a time, so
each change can be attributed on its own.

  3a (gated=False)
      Every candidate in the pool gets its own learned direction and is
      fully projected. The scope is the whole pool.
  3b (gated=True)
      3a plus a learned gate per direction and a sparsity penalty on the
      mean gate, so which components are used emerges from training.
      3a is 3b with every gate fixed at 1 and no penalty.
  3c (gated=False, align_weight > 0)
      3a plus a term pulling the directions together. With C the n x n
      matrix of pairwise cosines and J the matrix of ones,
          L_align = ||C - J||_F^2 / (n (n - 1)),
      the mean squared distance of each off-diagonal cosine from +1:
      0 when all directions are identical, 1 when orthogonal on average,
      4 for an antiparallel pair.
  3d (gated=True, align_weight > 0)
      3b plus 3c's term, with each pair weighted by both gates:
          L_align = sum_{i != j} (a_i a_j)^2 (cos_ij - 1)^2 / (n (n - 1)).
      A pair counts fully when both gates are open and drops out when
      either is closed. The gates are DETACHED in this term, so the
      optimiser cannot lower it by closing gates; which components are
      on is still decided by cross-entropy against sparsity, as in 3b.
      With all gates at 1 this is 3c's term; at align_weight = 0 the run
      is 3b.

OPERATOR. For a component's own write z (never the accumulated stream),
a learned unit direction m and a gate a in [0, 1]:

    z_new = (1 - a) z + a (I - m m^T) z     [intent: interpolate]
          = (I - a m m^T) z                 [folded into weights]
          = z - a (z . m) m                 [computed by hooks, O(d)]

The gate scales the rank-one correction, not m: scaling m would give
I - a^2 m m^T, so the gate would act squared. It also does not scale
the output, which would delete the whole write at a = 0.

TRAINING applies the operator with forward hooks; the weights are not
changed. fold_projector then writes it into the weights. The two are
equivalent because the operator does not depend on the input or token.

GRANULARITY (independent of the pool):
  "component": one (m, a) per candidate.
  "layer":     one (m, a) per layer. For heads, this equals removing m
               once from the layer's attention output projection.
  "shared":    one m for the whole pool, the limit align_weight -> inf.
               With gate_granularity="component" (3d), each head keeps
               its own gate.

BIASES. The attention output bias is shared by all heads in a layer, so
it belongs to no single head and is not projected. The MLP bias belongs
to one component and is projected. Hook and fold apply the same operator.
"""

import itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phase1 import _manual_per_head_attention
from phase2 import orthogonalize_head_write, orthogonalize_mlp_write

CLOSED_GATE_INIT = -3.0   # sigmoid(-3) = 0.047: every removal starts off
OPEN_GATE_INIT = 2.0      # sigmoid(+2) = 0.881: fallback when directions are random


def get_graph_dir(model_name, attack_type, dataset, base="results"):
    """Builds and returns results/{model_name}/{attack_type}/{dataset}."""
    path = Path(base) / model_name / attack_type / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def unit(v):
    """Normalises a vector to unit length, differentiably.

    Directions are learned as unconstrained vectors and normalised inside
    the forward pass rather than constrained in the optimiser. Because the
    operator depends only on the direction of v and not its length, the
    gradient with respect to v is automatically perpendicular to v, so the
    optimiser only ever rotates it and never needs its length corrected.
    """
    return v / (v.norm() + 1e-8)


def apply_removal(z, m, a):
    """Computes (I - a m m^T) z as z - a (z . m) m, for a batch of row vectors.

    Args:
        z: tensor whose last dimension is the model width, e.g. (B, N, d).
        m: unit direction, shape (d,).
        a: gate value in [0, 1], a float or a 0-dim tensor.
    """
    return z - a * (z @ m).unsqueeze(-1) * m


def alignment_loss(directions, gates=None):
    """Phase 3c's alignment term, ||C - J||_F^2 / (n (n - 1)), or, when gates
    are given, Phase 3d's gate-weighted term ||(C - J) * A||_F^2 / (n (n - 1)).

    C = M M^T is the n x n matrix of pairwise cosines between the n directions
    (rows of M, each normalised to unit length), with ones on its diagonal.
    J is the n x n matrix of ones, the cosine matrix of n identical directions.
    The squared Frobenius norm sums (cos_ij - 1)^2 over every entry; the
    diagonal contributes exactly 0, so dividing by the n (n - 1) off-diagonal
    entries makes this the MEAN squared deviation per pair. That keeps it on
    the same per-item scale as the cross-entropy terms (mean over images) and
    the sparsity term (mean over gates), independent of the pool size n.
    Per pair it is 0 for parallel, 1 for orthogonal and 4 for antiparallel
    directions, so a cosine of -1 is the worst case rather than a solution.

    Gates (Phase 3d). A = a a^T, so A_ij = a_i a_j, is multiplied into C - J
    entry by entry before the norm is taken, and the norm is squared, so pair
    (i, j) is weighted by (a_i a_j)^2: 1 when both removals are fully on, 0
    when either is off. A is built from the gate values DETACHED, so this term
    sends no gradient to any gate and can only rotate directions. The divisor
    stays the fixed n (n - 1), not the sum of A: dividing by the weights would
    undo the weighting whenever all gates are equally small, as they are at
    initialisation, and pull every direction at full strength from step 0.
    With every gate at 1 the gated term equals the ungated one exactly.

    Args:
        directions: list of (d,) tensors, normalised here.
        gates: optional sequence of n gate values in [0, 1] (floats or 0-dim
            tensors), one per direction and in the same order.

    Returns:
        0-dim tensor in [0, 4]; 0 for fewer than two directions.
    """
    M = torch.stack([unit(v) for v in directions])
    n = M.shape[0]
    if n < 2:
        return M.new_zeros(())
    D = M @ M.T - 1.0
    if gates is not None:
        a = torch.stack([torch.as_tensor(g, dtype=M.dtype, device=M.device) for g in gates]).detach()
        if a.numel() != n:
            raise ValueError(f"alignment_loss: {a.numel()} gates for {n} directions")
        D = D * torch.outer(a, a)
    return (D ** 2).sum() / (n * (n - 1))


def removal_matrix(m, a, device=None):
    """Builds I - a m m^T explicitly. Used only for folding into weights."""
    m = m.to(device) if device is not None else m
    return torch.eye(m.shape[0], device=m.device, dtype=m.dtype) - a * torch.outer(m, m)


def list_candidate_components(model, pool):
    """Lists the candidates in a pool, always across every layer.

    Args:
        pool: "heads", "mlps", or "both".

    Returns:
        list of (layer, component), component being "head_h" or "mlp".
        ViT-Small (L=12, H=6): 72, 12 and 84 candidates respectively.
    """
    n_layers = len(model.transformer)
    n_heads = model.transformer[0].attn.num_heads
    out = []
    if pool in ("heads", "both"):
        out += [(l, f"head_{h}") for l in range(n_layers) for h in range(n_heads)]
    if pool in ("mlps", "both"):
        out += [(l, "mlp") for l in range(n_layers)]
    return out


def parameter_key(candidate, granularity):
    """Maps a candidate to the key of the parameter group owning its direction and gate."""
    if granularity == "component":
        return candidate
    if granularity == "layer":
        return (candidate[0], "layer")
    if granularity == "shared":
        # One direction for the whole pool: the limit of Phase 3c's alignment
        # term as its weight grows without bound, where C = J exactly.
        return ("all", "shared")
    raise ValueError(f"granularity must be 'component', 'layer' or 'shared', got {granularity!r}")


def list_parameter_keys(candidates, granularity):
    """Returns the ordered, de-duplicated parameter keys for a pool at a granularity."""
    seen, keys = set(), []
    for c in candidates:
        pk = parameter_key(c, granularity)
        if pk not in seen:
            seen.add(pk)
            keys.append(pk)
    return keys


def dom_init_directions(head_dirs, mlp_dirs, candidates, granularity="component", cls_position=0,
                        sign_reference=None):
    """Builds difference-of-means initialisations from Phase 1's extracted directions.

    Starting each direction at its own difference-of-means direction makes
    step 0 a known configuration rather than an arbitrary one, which is what
    makes closed-gate initialisation safe. At layer granularity a group's
    initialisation is the sum of its members' directions, which by linearity
    of a mean difference is exactly that layer's own combined direction.

    A projector does not depend on the sign of its direction, since
    (I - m m^T) = (I - (-m)(-m)^T), so flipping an initial direction changes
    nothing about the removal at step 0. It does matter to Phase 3c's
    alignment term, which aims for cosine +1 and would otherwise be asked to
    undo an arbitrary sign convention. sign_reference (Phase 3c passes r_hat)
    flips each group's direction, after its members are summed, so that it
    points into the same half-space as the reference. Flipping after summing
    keeps every group's direction identical up to sign to what Phases 3a and
    3b start from, so step 0 is unchanged at every granularity. At
    granularity "shared" the group is the whole pool, and its direction is
    the sum of every member's own mean difference. None, the default, leaves
    the directions exactly as Phases 3a and 3b have always used them.

    Args:
        head_dirs: dict layer -> (H, N, d), from phase1.extract_component_activations.
        mlp_dirs: dict layer -> (N, d), same source.
        sign_reference: optional (d,) vector; see Sign above.

    Returns:
        dict parameter_key -> unit vector of shape (d,).
    """
    accum = {}
    for (l, c) in candidates:
        v = mlp_dirs[l][cls_position] if c == "mlp" else head_dirs[l][int(c.split("_")[1]), cls_position]
        pk = parameter_key((l, c), granularity)
        accum[pk] = accum.get(pk, 0) + v.detach().float().cpu()
    if sign_reference is not None:
        ref = sign_reference.detach().float().cpu()
        accum = {pk: (-v if torch.dot(v, ref) < 0 else v) for pk, v in accum.items()}
    return {pk: unit(v) for pk, v in accum.items()}


def _attn_hook(model, layer, head_entries, head_offsets=None):
    """Applies an optional offset and then the removal to every candidate head
    at this layer, in one hook. Heads absent from both dicts pass through unmodified.

    Args:
        head_entries: dict head index -> (m, a), the removal.
        head_offsets: dict head index -> (d,) tensor added before removal.
    """
    blk = model.transformer[layer]
    head_offsets = head_offsets or {}

    def hook(mod, inp, out):
        contribs, _ = _manual_per_head_attention(mod, inp[0])   # (B, H, N, d)
        summed = None
        for h in range(contribs.shape[1]):
            hc = contribs[:, h]
            if h in head_offsets:
                hc = hc + head_offsets[h]
            if h in head_entries:
                m, a = head_entries[h]
                hc = apply_removal(hc, m, a)
            summed = hc if summed is None else summed + hc
        if mod.out_proj.bias is not None:
            summed = summed + mod.out_proj.bias     # added after removal, never projected
        return (summed, out[1] if isinstance(out, tuple) else None)

    return blk.attn.register_forward_hook(hook)


def _mlp_hook(model, layer, m_a, offset=None):
    """Applies an optional offset then the removal to this layer's MLP output."""
    def hook(mod, inp, out):
        if offset is not None:
            out = out + offset
        return apply_removal(out, *m_a) if m_a is not None else out
    return model.transformer[layer].mlp.register_forward_hook(hook)


def register_removal_hooks(model, candidates, m_and_a=None, offsets=None):
    """Registers every hook a pool needs, for removal, for gradient offsets, or both.

    Args:
        m_and_a: dict (layer, component) -> (m, a), the removal. May be None.
        offsets: dict (layer, component) -> (d,) tensor, added to that
            component's write before removal. May be None.

    Returns:
        list of hook handles, to be removed by the caller.
    """
    by_layer = {}
    for (l, c) in candidates:
        e = by_layer.setdefault(l, {"heads": {}, "head_off": {}, "mlp": None, "mlp_off": None})
        rem = m_and_a.get((l, c)) if m_and_a else None
        off = offsets.get((l, c)) if offsets else None
        if c == "mlp":
            e["mlp"], e["mlp_off"] = rem, off
        else:
            h = int(c.split("_")[1])
            if rem is not None:
                e["heads"][h] = rem
            if off is not None:
                e["head_off"][h] = off

    hooks = []
    for l, e in by_layer.items():
        if e["heads"] or e["head_off"]:
            hooks.append(_attn_hook(model, l, e["heads"], e["head_off"]))
        if e["mlp"] is not None or e["mlp_off"] is not None:
            hooks.append(_mlp_hook(model, l, e["mlp"], e["mlp_off"]))
    return hooks


def train_projector(model, loader, candidates, gated=True, granularity="component",
                    init_directions=None, init_gates=None, gamma=0.25,
                    epochs=200, lr=1e-3, gate_lr=1e-2, patience=10, min_delta=1e-4,
                    max_batches=None, target_class=None, target_class_loader=None,
                    num_classes=None, target_class_weight=None, device="cuda",
                    verbose=True, print_every=10, align_weight=0.0, gate_granularity=None):
    """Trains one direction per parameter group (Phases 3a, 3c), or one
    direction and one gate (Phases 3b, 3d), jointly against one objective.

        3a:  L = CE(f(x_c), y) + CE(f(x_t), y) + mu * CE(f(x_c^t), t)
        3b:  3a + gamma * mean_s(a_s)
        3c:  3a + align_weight * ||C - J||_F^2 / (n (n - 1))
        3d:  3b + align_weight * ||(C - J) * A||_F^2 / (n (n - 1)),
             A = a a^T from the detached gates (see alignment_loss)

    The third term, on genuine target-class clean images, is used only when
    target_class_loader is given. The target class never appears in the
    first two terms.

    Sparsity. The penalty is the MEAN gate, so it lies in [0, gamma], on
    the same scale as the cross-entropy (at most about ln K), and gamma reads
    as the cross-entropy the run will pay to switch every removal off. A sum
    would push harder on larger pools, making pools of different size
    incomparable. gamma must be swept per configuration.

    Gate initialisation. With init_directions, gates start CLOSED at
    sigmoid(-3) = 0.047, so the penalty starts near zero, the run opens
    gates rather than pruning them, and step 0 is numerically the unmodified
    model (a free check on the hooks). With random directions, gates start
    OPEN at sigmoid(2) = 0.881, since closed gates and random directions
    deadlock: a direction's gradient is scaled by its gate, and a gate has
    no reason to open while its direction is meaningless.

    Args:
        model: the backdoored model, frozen.
        loader: paired loader yielding (x_c, y, x_t, y_t).
        candidates: from list_candidate_components.
        gated: True for Phases 3b and 3d, False for 3a and 3c.
        granularity: "component", "layer" or "shared", for the directions.
        init_directions: dict parameter_key -> (d,) unit vector, from
            dom_init_directions. Strongly recommended.
        init_gates: dict parameter_key -> gate logit, to continue a run.
        gamma: sparsity weight. Ignored when gated=False.
        target_class_loader: an unfiltered clean loader. The paired loader
            excludes clean images of the target class, so this term keeps
            the model predicting it correctly. Requires target_class.
        align_weight: weight lambda on the alignment term (see
            alignment_loss); gate-weighted when gated=True (Phase 3d).
            0, the default, reproduces Phases 3a and 3b exactly. When
            positive, early stopping tracks cross-entropy plus the alignment
            term (never the sparsity term), since alignment can keep
            improving while cross-entropy rises.
        gate_granularity: defaults to granularity. Only with
            granularity="shared" may it be "component", giving one direction
            for the pool and one gate per candidate (Phase 3d's shared run).

    Returns:
        dict with
          dir_by_key, gate_by_key: per candidate, replicated across tied
              groups so folding and evaluation work at any granularity;
          dir_by_param_key, gate_by_param_key: per group, what sparsity is
              reported over (gate groups differ from direction groups only
              in Phase 3d's shared run);
          gate_logits, gate_history, step0_ce, config;
          loss_history: cross-entropy only, comparable across variants;
          align_history, align_step0: Phases 3c and 3d only.
    """
    if target_class_loader is not None and target_class is None:
        raise ValueError("target_class_loader requires target_class, to filter its batches "
                         "down to genuine target-class images.")
    if target_class_weight is None:
        # Frequency matching: target-class images are 1/K of clean data, so weighting
        # them at 1/K reproduces their natural share
        target_class_weight = 1.0 / num_classes if num_classes else 1.0

    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gate_granularity = gate_granularity or granularity
    if gate_granularity != granularity and granularity != "shared":
        raise ValueError("gate_granularity may differ from granularity only when granularity='shared'")
    d = model.pos_embedding.shape[-1]
    param_keys = list_parameter_keys(candidates, granularity)
    pk_of = {c: parameter_key(c, granularity) for c in candidates}
    gate_keys = list_parameter_keys(candidates, gate_granularity) if gated else []
    gk_of = {c: parameter_key(c, gate_granularity) for c in candidates}
    if align_weight > 0 and len(param_keys) < 2:
        raise ValueError("align_weight has no effect with a single shared direction; pass 0")
    gate_init = CLOSED_GATE_INIT if init_directions is not None else OPEN_GATE_INIT

    dirs, gates = {}, {}
    for pk in param_keys:
        v = torch.randn(d, device=device) * 0.01
        if init_directions is not None and pk in init_directions:
            v = init_directions[pk].to(device=device, dtype=v.dtype).clone()
        dirs[pk] = torch.nn.Parameter(v)
    for gk in gate_keys:
        logit = init_gates[gk] if (init_gates is not None and gk in init_gates) else gate_init
        gates[gk] = torch.nn.Parameter(torch.tensor(float(logit), device=device))

    def _align():
        """The alignment term; gate-weighted (Phase 3d) when the run is gated.
        With alignment on, direction and gate groups coincide (shared runs
        have a single direction and are refused above), so gates[pk] exists."""
        a_s = [torch.sigmoid(gates[pk]) for pk in param_keys] if gated else None
        return alignment_loss([dirs[pk] for pk in param_keys], a_s)

    groups = [{"params": list(dirs.values()), "lr": lr}]
    if gated:
        groups.append({"params": list(gates.values()), "lr": gate_lr})
    opt = torch.optim.Adam(groups)

    tc_iter = itertools.cycle(target_class_loader) if target_class_loader is not None else None

    def _ce(x_c, y, x_t):
        m_and_a = {c: (unit(dirs[pk_of[c]]),
                       torch.sigmoid(gates[gk_of[c]]) if gated else 1.0) for c in candidates}
        hooks = register_removal_hooks(model, candidates, m_and_a)
        ce = F.cross_entropy(model(x_c), y)
        if tc_iter is not None:
            b = next(tc_iter)
            xi, yi = b[0].to(device), b[1].to(device)
            keep = yi == target_class
            if keep.any():
                ce = ce + target_class_weight * F.cross_entropy(model(xi[keep]), yi[keep])
        for h in hooks:
            h.remove()
        return ce

    with torch.no_grad():
        x_c, y, x_t, _ = next(iter(loader))
        step0_ce = _ce(x_c.to(device), y.to(device), x_t.to(device)).item()
        align_step0 = _align().item() if align_weight > 0 else None
    if verbose:
        variant = (("3d gated+aligned" if align_weight > 0 else "3b gated") if gated
                   else ("3c aligned" if align_weight > 0 else "3a ungated"))
        extra = f" | step-0 align {align_step0:.4f}" if align_weight > 0 else ""
        print(f"  {variant} | {granularity} | "
              f"{len(candidates)} candidates, {len(param_keys)} directions, {len(gate_keys)} gates | step-0 CE {step0_ce:.4f}{extra}")

    loss_history, gate_history, align_history = [], [], []
    best, stall, ran = float("inf"), 0, 0
    g_now = np.ones(len(gate_keys) or len(param_keys))

    for epoch in range(epochs):
        losses, aligns = [], []
        for i, (x_c, y, x_t, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            with torch.enable_grad():
                ce = _ce(x_c.to(device), y.to(device), x_t.to(device))
                if not ce.requires_grad:
                    raise RuntimeError(
                        "train_projector: the loss carries no gradient graph. The removal "
                        "hooks did not reach the model's output, which usually means the "
                        "model's forward pass is wrapped in torch.no_grad(), or that its "
                        "attention blocks are not at model.transformer[l].attn / .mlp as "
                        "the hooks assume.")
                loss = ce
                if gated:
                    mean_gate = torch.stack([torch.sigmoid(gates[gk]) for gk in gate_keys]).mean()
                    loss = loss + gamma * mean_gate
                if align_weight > 0:
                    align = _align()
                    loss = loss + align_weight * align
                    aligns.append(align.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(ce.item())

        ran = epoch + 1
        mean_ce = float(np.mean(losses)) if losses else 0.0
        loss_history.append(mean_ce)
        mean_align = float(np.mean(aligns)) if aligns else 0.0
        if align_weight > 0:
            align_history.append(mean_align)
        if gated:
            with torch.no_grad():
                g_now = np.array([torch.sigmoid(gates[gk]).item() for gk in gate_keys])
            gate_history.append(g_now)

        if verbose and (epoch == 0 or ran % print_every == 0):
            extra = f", active {g_now.sum():.2f}/{len(gate_keys)}" if gated else ""
            if align_weight > 0:
                extra += f", align {mean_align:.4f}"
            print(f"  epoch {ran}/{epochs}: CE {mean_ce:.4f}{extra}")

        # Phases 3a/3b stop on cross-entropy, as before. Phases 3c/3d add the alignment
        # term, since alignment can keep improving while cross-entropy rises; the
        # sparsity term stays excluded, as in 3b, so 3d at align_weight = 0 is 3b.
        criterion = mean_ce + align_weight * mean_align if align_weight > 0 else mean_ce
        if criterion < best - min_delta:
            best, stall = criterion, 0
        else:
            stall += 1
        if stall >= patience:
            break

    if was_training:
        model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        dir_pk = {pk: unit(dirs[pk].detach()).cpu() for pk in param_keys}
        if gated:
            gate_pk = {gk: torch.sigmoid(gates[gk].detach()).item() for gk in gate_keys}
            logit_pk = {gk: gates[gk].detach().item() for gk in gate_keys}
        else:
            gk_of = pk_of
            gate_pk = {pk: 1.0 for pk in param_keys}
            logit_pk = {pk: None for pk in param_keys}

    return {
        "dir_by_key": {c: dir_pk[pk_of[c]] for c in candidates},
        "gate_by_key": {c: gate_pk[gk_of[c]] for c in candidates},
        "dir_by_param_key": dir_pk,
        "gate_by_param_key": gate_pk,
        "gate_logits": logit_pk,
        "param_key_of": pk_of,
        "gate_key_of": gk_of,
        "loss_history": loss_history,
        "gate_history": gate_history,
        "step0_ce": step0_ce,
        "align_history": align_history,
        "align_step0": align_step0,
        "epochs_run": ran,
        "config": {"gated": gated, "granularity": granularity, "gamma": gamma if gated else 0.0,
                   "lr": lr, "gate_lr": gate_lr, "gate_init": gate_init if gated else None,
                   "n_candidates": len(candidates), "n_directions": len(param_keys),
                   "informed_init": init_directions is not None,
                   "align_weight": align_weight, "gate_granularity": gate_granularity if gated else None},
    }


@torch.no_grad()
def evaluate_projector(model, dir_by_key, gate_by_key, loader, target_class, device, max_batches=None):
    """Evaluates through the same hooks training used, at the learned gate values.

    Returns:
        dict with acc, asr, ra, n.
    """
    model.eval()
    candidates = list(dir_by_key.keys())
    hooks = register_removal_hooks(
        model, candidates, {c: (dir_by_key[c].to(device), gate_by_key[c]) for c in candidates})

    ok_c = ok_t = ok_ra = n = 0
    for i, (x_c, y, x_t, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x_c, y, x_t = x_c.to(device), y.to(device), x_t.to(device)
        p_c, p_t = model(x_c).argmax(1), model(x_t).argmax(1)
        ok_c += (p_c == y).sum().item()
        ok_t += (p_t == target_class).sum().item()
        ok_ra += (p_t == y).sum().item()
        n += len(x_c)
    for h in hooks:
        h.remove()
    return {"acc": ok_c / n, "asr": ok_t / n, "ra": ok_ra / n, "n": n} if n else \
           {"acc": 0.0, "asr": 0.0, "ra": 0.0, "n": 0}


@torch.no_grad()
def check_target_class_suppression(model, loader, target_class, device, max_batches=None):
    """Checks whether the target class was suppressed outright rather than defended.

    The triggered term can be reduced either by removing the trigger's causal
    influence, which is the intent, or by making the model stop predicting the
    target class at all, which is not. Overall clean accuracy barely
    distinguishes them, since the target class is about 1/K of the data.

    Run on the FOLDED model, with no hooks, since that is what is deployed.

    Args:
        loader: an unfiltered clean loader, so genuine target-class images
            actually appear in it.

    Returns:
        dict with target_class_clean_acc, frac_predicted_target_class
        (expected near 1/K if not suppressed), n_target_class_seen.
    """
    model.eval()
    ok_t = n_t = pred_t = n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        p = model(x).argmax(1)
        keep = y == target_class
        ok_t += (p[keep] == target_class).sum().item()
        n_t += keep.sum().item()
        pred_t += (p == target_class).sum().item()
        n += len(x)
    return {"target_class_clean_acc": ok_t / n_t if n_t else None,
            "frac_predicted_target_class": pred_t / n if n else 0.0,
            "n_target_class_seen": n_t}


def fold_projector(model, dir_by_key, gate_by_key, device):
    """Writes every component's removal permanently into its weights.

    Uses the exact operator training used, I - a m m^T, at every candidate. A
    gate of 0.3 folds as three-tenths of a removal, so no threshold
    is applied and no component is discarded: the folded model reproduces the
    activation-level intervention exactly, for every gate value. In Phase 3a
    every gate is 1 and this reduces to Phase 2's write-side orthogonalisation
    with a learned direction.

    Returns:
        a deep copy of model with every removal folded in.
    """
    model = deepcopy(model)
    for (l, c) in dir_by_key:
        P = removal_matrix(dir_by_key[(l, c)].to(device), gate_by_key[(l, c)])
        blk = model.transformer[l]
        if c == "mlp":
            orthogonalize_mlp_write(blk.mlp, P)
        else:
            orthogonalize_head_write(blk.attn, int(c.split("_")[1]), P)
    return model


def summarise_gates(gate_by_param_key, n_bins=10):
    """Summarises how strongly each parameter group's removal is applied.

    expected_active is the sum of the gates: the effective number of removals in
    use, counting a gate of 0.5 as half a removal. The histogram shows whether
    the run concentrated its removal on a few components or spread it thinly.

    Reported over parameter groups, not candidates, since one gate at layer
    granularity controls a whole layer.
    """
    g = np.array(list(gate_by_param_key.values()), dtype=float)
    if g.size == 0:
        return {}
    counts, edges = np.histogram(g, bins=n_bins, range=(0.0, 1.0))
    return {"n_directions": int(g.size), "expected_active": float(g.sum()),
            "mean_gate": float(g.mean()), "min": float(g.min()), "max": float(g.max()),
            "median": float(np.median(g)), "histogram": counts.tolist(),
            "bin_edges": edges.tolist(),
            "ranked": sorted(((k, float(v)) for k, v in gate_by_param_key.items()),
                             key=lambda kv: -kv[1])}


def compare_rankings(scores_a, scores_b, top_k=10):
    """Compares two rankings over the same components.
 
    Can be used to check whether the components Phase 3b gates
    most strongly are the ones Phase 1 identified causally, and whether the same
    components come out on top across scopes, seeds and attacks. Reporting the
    overlap matters more than counting how many gates are large, because a run
    can open many gates and still disagree entirely about which ones.
 
    Args:
        scores_a, scores_b: dicts over component keys. Only keys present in both
            are compared. Higher is more important in each.
        top_k: size of the two head-to-head sets.
 
    Returns:
        dict with n_shared, spearman (rank correlation over all shared keys),
        overlap (shared members of the two top_k sets), and the two top lists.
    """
    keys = [k for k in scores_a if k in scores_b]
    if len(keys) < 2:
        return {}
    a = np.array([scores_a[k] for k in keys], dtype=float)
    b = np.array([scores_b[k] for k in keys], dtype=float)
 
    def ranks(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v))
        return r
 
    k = min(top_k, len(keys))
    top_a = {keys[i] for i in np.argsort(-a)[:k]}
    top_b = {keys[i] for i in np.argsort(-b)[:k]}
    sp = float(np.corrcoef(ranks(a), ranks(b))[0, 1])
    return {"n_shared": len(keys), "spearman": sp, "overlap": len(top_a & top_b), "top_k": k,
            "top_a": sorted(top_a, key=lambda x: -scores_a[x]),
            "top_b": sorted(top_b, key=lambda x: -scores_b[x])}


@torch.no_grad()
def analyse_learned_directions(model, dir_by_param_key, r_hat=None, dom_by_param_key=None):
    """Measures how far each learned direction moved from the fixed directions it
    could have been, which is what says what the learning added.

    cos_r_hat and cos_dom are cosines against Phase 0's global direction and
    against the group's own difference-of-means direction (its initialisation).

    write_space_norm, for heads only, is the fraction of the learned direction's
    norm lying inside that head's own write space col(W_O^{l,h}), of dimension at
    most d_h. A difference-of-means direction is a difference of that head's own
    contributions and therefore lies entirely inside this space by construction,
    so this equals 1 for it. A learned direction is under no such constraint, so
    this number says directly whether the optimiser used freedom the difference
    of means never had.
    """
    out = {}
    for pk, v in dir_by_param_key.items():
        m = unit(v.detach().float().cpu())
        e = {}
        if r_hat is not None:
            e["cos_r_hat"] = float(torch.dot(m, unit(r_hat.detach().float().cpu())))
        if dom_by_param_key is not None and pk in dom_by_param_key:
            e["cos_dom"] = float(torch.dot(m, unit(dom_by_param_key[pk].detach().float().cpu())))
        l, c = pk
        if isinstance(c, str) and c.startswith("head_"):
            attn = model.transformer[l].attn
            hd = attn.embed_dim // attn.num_heads
            h = int(c.split("_")[1])
            Q, _ = torch.linalg.qr(attn.out_proj.weight.data[:, h * hd:(h + 1) * hd].float().cpu(),
                                   mode="reduced")
            e["write_space_norm"] = float((Q.T @ m).norm())
        out[pk] = e
    return out


@torch.no_grad()
def alignment_summary(dir_by_param_key, sign_reference=None, compare_to=None, n_bins=20,
                      gate_by_param_key=None, open_at=0.5):
    """Summarises how far a set of learned directions agree with each other.

    This is the quantity Phase 3c's alignment term acts on, measured after
    training, so it can be reported for any variant: for Phase 3a and 3b it
    describes how far apart unconstrained directions lie, for Phase 3c how far
    the term succeeded in bringing them together.

    Args:
        dir_by_param_key: dict parameter_key -> (d,) direction.
        sign_reference: optional (d,) vector. Each direction is first flipped
            into the reference's half-space, the convention Phase 3c starts
            from. Use it for unconstrained directions, whose signs carry no
            meaning; leave it None for Phase 3c, whose signs were trained.
        compare_to: optional (d,) vector (e.g. r_hat) to compare the consensus
            direction with.
        gate_by_param_key: optional gates over the same keys (Phases 3b, 3d).
            Adds gated_align_loss (Phase 3d's term), weighted_mean_cos (the
            mean off-diagonal cosine weighted by (a_i a_j)^2, the term's own
            pair weights) and "open": the same statistics restricted to the
            directions whose gate is at least open_at, i.e. those actually
            doing the removing. A closed gate leaves its direction almost
            untrained, so agreement over all pairs understates it.

    Returns:
        dict with n; the signed off-diagonal cosines' mean, median, quartiles,
        min and max; the median and mean absolute cosine, which do not depend
        on any sign convention; the share of pairs above 0.9;
        align_loss (the alignment term's value); mean_norm, the length of the
        mean of the unit directions, which is 1 only if all are identical and
        satisfies mean_norm^2 = mean of every entry of C; the consensus (the
        mean direction, normalised); cos_consensus (with compare_to); and a
        histogram of the off-diagonal cosines over [-1, 1].
    """
    keys = list(dir_by_param_key)
    M = torch.stack([unit(dir_by_param_key[k].detach().float().cpu()) for k in keys])
    if sign_reference is not None:
        ref = unit(sign_reference.detach().float().cpu())
        flip = 1.0 - 2.0 * (M @ ref < 0).float()
        M = M * flip[:, None]
    n = M.shape[0]
    mean_dir = M.mean(0)
    out = {"n": n, "mean_norm": float(mean_dir.norm()), "consensus": unit(mean_dir)}
    if compare_to is not None:
        out["cos_consensus"] = float(torch.dot(unit(mean_dir), unit(compare_to.detach().float().cpu())))
    if n < 2:
        return out
    C = M @ M.T
    iu = torch.triu_indices(n, n, 1)
    c = C[iu[0], iu[1]].numpy()
    out.update({
        "mean": float(c.mean()), "median": float(np.median(c)),
        "q25": float(np.quantile(c, 0.25)), "q75": float(np.quantile(c, 0.75)),
        "min": float(c.min()), "max": float(c.max()),
        "abs_median": float(np.median(np.abs(c))), "abs_mean": float(np.abs(c).mean()),
        "frac_above_0_9": float((c > 0.9).mean()),
        "align_loss": float(((C - 1.0) ** 2).sum() / (n * (n - 1))),
        "histogram": np.histogram(c, bins=n_bins, range=(-1.0, 1.0))[0].tolist(),
    })
    if gate_by_param_key is not None and all(k in gate_by_param_key for k in keys):
        a = torch.tensor([float(gate_by_param_key[k]) for k in keys])
        A = torch.outer(a, a)
        off = ~torch.eye(n, dtype=torch.bool)
        w = (A ** 2)[off]
        out["gated_align_loss"] = float((((C - 1.0) * A) ** 2).sum() / (n * (n - 1)))
        out["weighted_mean_cos"] = float((w * C[off]).sum() / w.sum()) if w.sum() > 0 else float("nan")
        idx = torch.nonzero(a >= open_at).flatten()
        op = {"n": int(idx.numel()), "open_at": open_at}
        if idx.numel() >= 1:
            Mo = M[idx]
            op["mean_norm"] = float(Mo.mean(0).norm())
            op["consensus"] = unit(Mo.mean(0))
            if compare_to is not None:
                op["cos_consensus"] = float(torch.dot(op["consensus"], unit(compare_to.detach().float().cpu())))
        if idx.numel() >= 2:
            Co = Mo @ Mo.T
            iu = torch.triu_indices(len(idx), len(idx), 1)
            co = Co[iu[0], iu[1]].numpy()
            op.update({"mean": float(co.mean()), "median": float(np.median(co)),
                       "q25": float(np.quantile(co, 0.25)), "q75": float(np.quantile(co, 0.75)),
                       "min": float(co.min()), "max": float(co.max()),
                       "abs_median": float(np.median(np.abs(co))), "abs_mean": float(np.abs(co).mean()),
                       "frac_above_0_9": float((co > 0.9).mean()),
                       "histogram": np.histogram(co, bins=n_bins, range=(-1.0, 1.0))[0].tolist()})
        out["open"] = op
    return out