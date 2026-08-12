"""
Phase 1: Exploratory Circuit Analysis for backdoor detection in ViTs.
Covers proposal Sections 4.1.2 to 4.1.5: layer-level analysis (logit lens),
component-level decomposition, activation patching, path patching, functional
role classification, steering, and a Phase 2 defense preview.

Assumes phase0.py has already run and produced: a backdoor-trained model,
directions_cls (r^l for every layer), directions_all (all-token variant),
and r_hat (the causally-validated most-representative-layer direction).

All functions take the same 4-output paired loader as phase0
(clean_img, clean_label, backdoor_img, backdoor_label) and the same
BaseVisionTransformer architecture (ViT-Small or DeiT-Tiny).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path
from tqdm import tqdm
from itertools import product, combinations

GRAPH_DIR = Path("./results/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Shared helpers
# ===========================================================================
def _manual_per_head_attention(attn_module, x):
    """
    Reimplements nn.MultiheadAttention's math by hand so each head's own
    contribution can be isolated (the built-in module only exposes the
    summed output across heads).
    Returns head_contribs (B,H,N,D): each head's own full-D write to the
    residual stream, before summing across heads and before out_proj.bias.
    Also returns attn_weights (B,H,N,N), the softmax attention pattern.
    """
    D = attn_module.embed_dim
    H = attn_module.num_heads
    hd = D // H
    W = attn_module.in_proj_weight
    Wq, Wk, Wv = W[:D], W[D:2 * D], W[2 * D:]
    b = attn_module.in_proj_bias
    bq, bk, bv = (b[:D], b[D:2 * D], b[2 * D:]) if b is not None else (None, None, None)

    q = F.linear(x, Wq, bq)
    k = F.linear(x, Wk, bk)
    v = F.linear(x, Wv, bv)
    B, N, _ = x.shape
    q = q.view(B, N, H, hd).transpose(1, 2)
    k = k.view(B, N, H, hd).transpose(1, 2)
    v = v.view(B, N, H, hd).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    head_out = attn_weights @ v

    Wo = attn_module.out_proj.weight
    contribs = [head_out[:, h] @ Wo[:, h * hd:(h + 1) * hd].T for h in range(H)]
    return torch.stack(contribs, dim=1), attn_weights


@torch.no_grad()
def sanity_check_manual_attention(model, loader, device, atol=1e-4):
    """
    One-time correctness check: confirms _manual_per_head_attention's
    per-head decomposition sums back to the model's real attention output.
    Run this first. If it fails, no patching result in this file can be
    trusted, since every technique below depends on this reconstruction.
    """
    model.eval()
    img_c, _, _, _ = next(iter(loader))
    img_c = img_c.to(device)
    captured = {}

    def _capture(mod, inp, out, idx):
        captured[idx] = (inp[0].detach(), out[0].detach())

    hooks = [blk.attn.register_forward_hook(lambda m, i, o, idx=i_: _capture(m, i, o, idx))
             for i_, blk in enumerate(model.transformer)]
    model(img_c)
    for h in hooks:
        h.remove()

    max_diff = 0.0
    for l, (xn, true_out) in captured.items():
        head_contribs, _ = _manual_per_head_attention(model.transformer[l].attn, xn)
        recon = head_contribs.sum(dim=1)
        if model.transformer[l].attn.out_proj.bias is not None:
            recon = recon + model.transformer[l].attn.out_proj.bias
        max_diff = max(max_diff, (recon - true_out).abs().max().item())

    ok = max_diff < atol
    print(f"Sanity check: manual vs built-in attention max diff = {max_diff:.2e} "
          f"({'PASSED' if ok else 'FAILED, fix before trusting any patching result'})")
    return ok


def compute_trigger_token_position(image_size=32, patch_size=4, corner="bottom_right", cls_offset=1):
    """Fixed trigger token index for a static patch attack (BadNets only)."""
    grid = image_size // patch_size
    patch_idx = grid * grid - 1 if corner == "bottom_right" else 0
    return patch_idx + cls_offset


def detect_trigger_patch_positions(img_clean, img_bd, patch_size, cls_offset=1, top_frac=0.05, min_positions=1):
    """
    Finds which patch token(s) differ most between a clean/triggered pair,
    per sample. Works for both a fixed trigger (BadNets) and a moving one
    (IAB), since it does not assume a known location.
    """
    B, C, H, W = img_clean.shape
    grid = H // patch_size
    n_patches = grid * grid
    diff = (img_bd - img_clean).abs()
    patches = diff.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, n_patches, -1)
    patch_scores = patches.mean(dim=-1)
    n_flag = max(min_positions, int(np.ceil(top_frac * n_patches)))
    top_idx = patch_scores.topk(n_flag, dim=1).indices
    return [[int(p) + cls_offset for p in row] for row in top_idx]


# ===========================================================================
# Component-level activation analysis
# ===========================================================================
@torch.no_grad()
def extract_component_activations(model, loader, device, max_batches=None):
    """
    Computes the mean triggered-minus-clean difference for every attention
    head and MLP block, at every token position, by hooking each block's
    attention input and MLP output across the full loader.
    Returns head_dirs[l]: (H,N,D), mlp_dirs[l]: (N,D).
    """
    model.eval()
    n_layers = len(model.transformer)
    head_sum, mlp_sum, count = {}, {}, 0
    ln_inputs, mlp_outputs = {}, {}

    def _attn_in_hook(mod, inp, out, idx):
        ln_inputs[idx] = inp[0].detach()

    def _mlp_out_hook(mod, inp, out, idx):
        mlp_outputs[idx] = out.detach()

    hooks = []
    for i, blk in enumerate(model.transformer):
        hooks.append(blk.attn.register_forward_hook(lambda m, i_, o, idx=i: _attn_in_hook(m, i_, o, idx)))
        hooks.append(blk.mlp.register_forward_hook(lambda m, i_, o, idx=i: _mlp_out_hook(m, i_, o, idx)))

    for b_idx, (img_c, _, img_b, _) in enumerate(tqdm(loader, desc="Extracting component activations")):
        if max_batches and b_idx >= max_batches:
            break
        batch_size = img_c.size(0)
        model(img_c.to(device))
        clean_head, clean_mlp = {}, {}
        for l in range(n_layers):
            hc, _ = _manual_per_head_attention(model.transformer[l].attn, ln_inputs[l])
            clean_head[l] = hc.cpu()
            clean_mlp[l] = mlp_outputs[l].cpu()

        model(img_b.to(device))
        for l in range(n_layers):
            hc, _ = _manual_per_head_attention(model.transformer[l].attn, ln_inputs[l])
            head_sum[l] = head_sum.get(l, 0) + (hc.cpu() - clean_head[l]).sum(0)
            mlp_sum[l] = mlp_sum.get(l, 0) + (mlp_outputs[l].cpu() - clean_mlp[l]).sum(0)
        count += batch_size

    for h in hooks:
        h.remove()
    return {l: head_sum[l] / count for l in head_sum}, {l: mlp_sum[l] / count for l in mlp_sum}


def verify_extraction_consistency(head_dirs, mlp_dirs, directions_cls, cls_position=0, tol=1e-3):
    """
    Vector-arithmetic sanity check: r^l - r^(l-1) should equal the sum of
    every head's and the MLP's own write at that layer, since the residual
    stream is a pure additive accumulator. This holds by construction given
    correct hooks, so it is not a finding about the backdoor, only a bug
    check for a mismatch between phase0 and phase1's extraction.
    """
    max_residual = 0.0
    for l in range(len(head_dirs)):
        component_sum = head_dirs[l][:, cls_position, :].sum(0) + mlp_dirs[l][cls_position]
        r_l = directions_cls[l]
        r_lm1 = directions_cls[l - 1] if l > 0 else torch.zeros_like(r_l)
        own_write = r_l - r_lm1
        residual = (own_write - component_sum).norm().item() / (own_write.norm().item() + 1e-8)
        max_residual = max(max_residual, residual)
    ok = max_residual < tol
    print(f"Extraction consistency: max residual = {max_residual:.2e} "
          f"({'PASSED' if ok else 'FAILED, check phase0/phase1 sample mismatch'})")
    return ok


def rank_components_by_cosine_similarity(head_dirs, mlp_dirs, r_hat, cls_position=0, top_k=8, plot=True, save_dir=GRAPH_DIR):
    """Ranks every component by cosine similarity of its own CLS-position write against r_hat."""
    r = r_hat / (r_hat.norm() + 1e-8)
    scored = []
    for l in head_dirs:
        for h in range(head_dirs[l].shape[0]):
            v = head_dirs[l][h, cls_position, :]
            sim = F.cosine_similarity(v.unsqueeze(0), r.unsqueeze(0)).item()
            scored.append({"layer": l, "component": f"head_{h}", "cosine_sim": sim})
        v = mlp_dirs[l][cls_position, :]
        sim = F.cosine_similarity(v.unsqueeze(0), r.unsqueeze(0)).item()
        scored.append({"layer": l, "component": "mlp", "cosine_sim": sim})
    scored.sort(key=lambda d: abs(d["cosine_sim"]), reverse=True)
    top = scored[:top_k]

    if plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [f"L{d['layer']}-{d['component']}" for d in top]
        vals = [d["cosine_sim"] for d in top]
        colors = ["crimson" if v > 0 else "steelblue" for v in vals]
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        ax.set_xlabel("Cosine similarity with r_hat")
        ax.set_title(f"Top-{top_k} components ranked by cosine similarity with r_hat")
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout(); plt.savefig(save_dir / "cosine_ranking_topk.png", dpi=150); plt.close()
    return top, scored


def select_top_component_per_layer(all_scored, top_per_layer=1):
    """Per-layer selection so every layer contributes a candidate (top_per_layer=None returns every component)."""
    by_layer = {}
    for d in all_scored:
        by_layer.setdefault(d["layer"], []).append(d)
    result = []
    for l in sorted(by_layer):
        layer_components = sorted(by_layer[l], key=lambda d: abs(d["cosine_sim"]), reverse=True)
        result.extend(layer_components if top_per_layer is None else layer_components[:top_per_layer])
    return result


def plot_cosine_similarity_distribution(all_scored, top_k, save_dir=GRAPH_DIR):
    """Plots |cosine sim| with r_hat for every component, sorted, to check whether top_k captures a distinct elite subset."""
    sorted_scores = sorted(all_scored, key=lambda d: abs(d["cosine_sim"]), reverse=True)
    vals = [abs(d["cosine_sim"]) for d in sorted_scores]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(vals) + 1), vals, "o-", color="steelblue", markersize=3)
    ax.axvline(top_k, color="crimson", ls="--", label=f"top_k = {top_k}")
    ax.set_xlabel("Component rank (all components)")
    ax.set_ylabel("|Cosine similarity| with r_hat")
    ax.set_title("Cosine similarity with r_hat across every component")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_dir / "cosine_ranking_all_components.png", dpi=150); plt.close()


# ===========================================================================
# Activation patching
# ===========================================================================
@torch.no_grad()
def _cache_batch_components(model, img, device):
    """One forward pass, caching every layer's per-head contributions (B,H,N,D) and MLP output (B,N,D)."""
    ln_inputs, mlp_outputs = {}, {}

    def _attn_in_hook(mod, inp, out, idx):
        ln_inputs[idx] = inp[0].detach()

    def _mlp_out_hook(mod, inp, out, idx):
        mlp_outputs[idx] = out.detach()

    hooks = []
    for i, blk in enumerate(model.transformer):
        hooks.append(blk.attn.register_forward_hook(lambda m, i_, o, idx=i: _attn_in_hook(m, i_, o, idx)))
        hooks.append(blk.mlp.register_forward_hook(lambda m, i_, o, idx=i: _mlp_out_hook(m, i_, o, idx)))
    model(img.to(device))
    for h in hooks:
        h.remove()

    head_cache, mlp_cache = {}, {}
    for l in range(len(model.transformer)):
        head_contribs, _ = _manual_per_head_attention(model.transformer[l].attn, ln_inputs[l])
        head_cache[l] = head_contribs
        mlp_cache[l] = mlp_outputs[l]
    return head_cache, mlp_cache


def _make_attn_patch_hook(target_head, position, source_head_cache_l, all_positions=False):
    """Swaps target_head's cached value into the live forward pass, at one position or every position."""
    def hook(module, inp, out):
        head_contribs, _ = _manual_per_head_attention(module, inp[0])
        head_contribs = head_contribs.clone()
        if all_positions:
            head_contribs[:, target_head, :, :] = source_head_cache_l[:, target_head, :, :]
        else:
            head_contribs[:, target_head, position, :] = source_head_cache_l[:, target_head, position, :]
        patched = head_contribs.sum(dim=1)
        if module.out_proj.bias is not None:
            patched = patched + module.out_proj.bias
        return (patched, out[1] if isinstance(out, tuple) else None)
    return hook


def _make_mlp_patch_hook(position, source_mlp_cache_l, all_positions=False):
    """Swaps the MLP's cached output into the live forward pass, at one position or every position."""
    def hook(module, inp, out):
        if all_positions:
            return source_mlp_cache_l.clone()
        patched = out.clone()
        patched[:, position, :] = source_mlp_cache_l[:, position, :]
        return patched
    return hook


def activation_patching(model, loader, top_components, target_class, device, positions=None, n_tokens=None,
                         max_batches=None, heatmap_filename="activation_patching_topk.png",
                         shared_abs_max=None, title_suffix="", save_dir=GRAPH_DIR):
    """
    Sufficiency test: swaps ONE component's triggered value into an
    otherwise-clean forward pass, at every token position by default, and
    measures the resulting shift in the target-class logit. This is the
    causal proxy equation Delta = L_t(f(x_c; h <- h(x_t))) - L_t(f(x_c)).
    Returns (attribution, stderr, abs_max). attribution/stderr are dicts
    keyed (layer, component, position) -> mean / standard error over
    per-sample deltas.
    """
    if n_tokens is None:
        n_tokens = model.pos_embedding.shape[1]
    if positions is None:
        positions = list(range(n_tokens))
    component_list = [(d["layer"], d["component"]) for d in top_components]
    results = {(l, c, p): [] for (l, c) in component_list for p in positions}

    for b_idx, (img_c, _, img_b, _) in enumerate(tqdm(loader, desc="Activation patching")):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        head_cache, mlp_cache = _cache_batch_components(model, img_b, device)
        with torch.no_grad():
            baseline_target = model(img_c)[:, target_class]
            for (l, comp) in component_list:
                blk = model.transformer[l]
                for p in positions:
                    if comp.startswith("head_"):
                        h = int(comp.split("_")[1])
                        hook = blk.attn.register_forward_hook(_make_attn_patch_hook(h, p, head_cache[l]))
                    else:
                        hook = blk.mlp.register_forward_hook(_make_mlp_patch_hook(p, mlp_cache[l]))
                    delta = (model(img_c)[:, target_class] - baseline_target).detach().cpu().tolist()
                    hook.remove()
                    results[(l, comp, p)].extend(delta)

    attribution = {k: float(np.mean(v)) for k, v in results.items()}
    stderr = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0 for k, v in results.items()}
    abs_max = _plot_activation_patching_heatmap(attribution, component_list, positions, save_dir,
                                                 heatmap_filename, shared_abs_max, title_suffix)
    return attribution, stderr, abs_max


def run_full_depth_activation_patching(model, loader, all_scored, target_class, device, max_batches=None, save_dir=GRAPH_DIR):
    """Sufficiency test for every component in the model (not just top-K), CLS position only, plotted as a compact grid."""
    all_components = select_top_component_per_layer(all_scored, top_per_layer=None)
    component_list = [(d["layer"], d["component"]) for d in all_components]
    results = {(l, c): [] for (l, c) in component_list}

    for b_idx, (img_c, _, img_b, _) in enumerate(tqdm(loader, desc="Full-depth activation patching")):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        head_cache, mlp_cache = _cache_batch_components(model, img_b, device)
        with torch.no_grad():
            baseline_target = model(img_c)[:, target_class]
            for (l, comp) in component_list:
                blk = model.transformer[l]
                if comp.startswith("head_"):
                    h = int(comp.split("_")[1])
                    hook = blk.attn.register_forward_hook(_make_attn_patch_hook(h, 0, head_cache[l]))
                else:
                    hook = blk.mlp.register_forward_hook(_make_mlp_patch_hook(0, mlp_cache[l]))
                delta = (model(img_c)[:, target_class] - baseline_target).mean().item()
                hook.remove()
                results[(l, comp)].append(delta)

    attribution = {k: float(np.mean(v)) for k, v in results.items()}
    _plot_full_depth_grid(attribution, all_components, save_dir)
    return attribution, all_components


def _plot_full_depth_grid(attribution, all_components, save_dir=GRAPH_DIR):
    """Compact grid (x = layer, y = component slot) so the plot stays a fixed height regardless of model depth."""
    layers = sorted(set(d["layer"] for d in all_components))
    head_slots = sorted({d["component"] for d in all_components if d["component"].startswith("head_")},
                         key=lambda s: int(s.split("_")[1]))
    slots = head_slots + (["mlp"] if any(d["component"] == "mlp" for d in all_components) else [])
    mat = np.full((len(slots), len(layers)), np.nan)
    for d in all_components:
        mat[slots.index(d["component"]), layers.index(d["layer"])] = attribution.get((d["layer"], d["component"]), np.nan)

    abs_max = np.nanmax(np.abs(mat)) + 1e-8
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(layers) + 3), max(4, 0.4 * len(slots) + 2)))
    norm = SymLogNorm(linthresh=max(abs_max * 0.01, 1e-3), vmin=-abs_max, vmax=abs_max)
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", norm=norm)
    ax.set_xticks(range(len(layers))); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_yticks(range(len(slots))); ax.set_yticklabels(slots, fontsize=8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Component slot")
    if len(layers) * len(slots) <= 400:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    plt.colorbar(im, ax=ax, label="Delta target-class logit (CLS position)")
    ax.set_title("Full-depth activation patching: every component, CLS position")
    plt.tight_layout(); plt.savefig(save_dir / "activation_patching_full_depth.png", dpi=150); plt.close()


def identify_strongest_position_per_component(attribution, component_list, positions):
    """For each component, returns the token position with the largest |Delta|, the empirically identified trigger position."""
    return {(l, c): max(((p, attribution[(l, c, p)]) for p in positions), key=lambda t: abs(t[1]))
            for (l, c) in component_list}


def _joint_necessity_eval(model, loader, component_set, target_class, device, position_mode="cls", position=0, max_batches=None):
    """
    Reverts every component in component_set to its clean value
    simultaneously, inside an otherwise-triggered forward pass.
    position_mode "cls" reverts only `position`, "all" reverts every position.
    Returns (asr, ra) as fractions.
    """
    target_hits = true_hits = total = 0
    for b_idx, (img_c, lbl_c, img_b, lbl_b) in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        clean_head_cache, clean_mlp_cache = _cache_batch_components(model, img_c, device)
        hooks = []
        for (l, comp) in component_set:
            blk = model.transformer[l]
            all_pos = position_mode == "all"
            if comp.startswith("head_"):
                h = int(comp.split("_")[1])
                hooks.append(blk.attn.register_forward_hook(_make_attn_patch_hook(h, position, clean_head_cache[l], all_pos)))
            else:
                hooks.append(blk.mlp.register_forward_hook(_make_mlp_patch_hook(position, clean_mlp_cache[l], all_pos)))
        with torch.no_grad():
            preds = model(img_b).argmax(1).cpu()
        for hook in hooks:
            hook.remove()
        target_hits += (preds == target_class).sum().item()
        true_hits += (preds == lbl_c).sum().item()
        total += len(img_c)
    return (target_hits / total if total else 0.0, true_hits / total if total else 0.0)


def necessity_patching(model, loader, top_components, target_class, device, max_batches=None, save_dir=GRAPH_DIR):
    """
    Necessity test, the reverse of activation_patching: reverts ONE
    component back to clean inside an otherwise-triggered run, and checks
    whether the backdoor still fires. Tests every top-K component
    individually, in both CLS-only and all-position modes.
    """
    component_list = [(d["layer"], d["component"]) for d in top_components]
    results = {(l, c, mode): {"target_hits": 0, "true_hits": 0, "total": 0}
               for (l, c) in component_list for mode in ("cls", "all")}

    for b_idx, (img_c, lbl_c, img_b, lbl_b) in enumerate(tqdm(loader, desc="Necessity patching")):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        clean_head_cache, clean_mlp_cache = _cache_batch_components(model, img_c, device)
        with torch.no_grad():
            for (l, comp) in component_list:
                blk = model.transformer[l]
                for mode in ("cls", "all"):
                    all_pos = mode == "all"
                    if comp.startswith("head_"):
                        h = int(comp.split("_")[1])
                        hook = blk.attn.register_forward_hook(_make_attn_patch_hook(h, 0, clean_head_cache[l], all_pos))
                    else:
                        hook = blk.mlp.register_forward_hook(_make_mlp_patch_hook(0, clean_mlp_cache[l], all_pos))
                    preds = model(img_b).argmax(1).cpu()
                    hook.remove()
                    r = results[(l, comp, mode)]
                    r["target_hits"] += (preds == target_class).sum().item()
                    r["true_hits"] += (preds == lbl_c).sum().item()
                    r["total"] += len(img_c)

    def _se(p, n):
        return float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0

    necessity = {}
    for k, v in results.items():
        n = v["total"]
        asr, ra = v["target_hits"] / n if n else 0.0, v["true_hits"] / n if n else 0.0
        necessity[k] = {"asr": asr, "ra": ra, "asr_se": _se(asr, n), "ra_se": _se(ra, n), "n": n}
    _plot_necessity_patching(necessity, component_list, save_dir)
    return necessity


def _plot_necessity_patching(necessity, component_list, save_dir=GRAPH_DIR):
    """Two panels, CLS-only and all-position, ASR/RA with binomial standard error bars for reverting one component alone."""
    labels = [f"L{l}-{c}" for (l, c) in component_list]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(labels) * 0.9) * 2, 5))
    for ax, mode, title in zip(axes, ["cls", "all"], ["CLS position only", "Every position"]):
        asr = [necessity[(l, c, mode)]["asr"] for (l, c) in component_list]
        ra = [necessity[(l, c, mode)]["ra"] for (l, c) in component_list]
        asr_se = [necessity[(l, c, mode)]["asr_se"] for (l, c) in component_list]
        ra_se = [necessity[(l, c, mode)]["ra_se"] for (l, c) in component_list]
        x = np.arange(len(labels)); width = 0.35
        ax.bar(x - width / 2, asr, width, yerr=asr_se, capsize=3, label="ASR", color="#d62728")
        ax.bar(x + width / 2, ra, width, yerr=ra_se, capsize=3, label="RA", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Necessity patching, {title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(save_dir / "necessity_patching.png", dpi=150); plt.close()


# ===========================================================================
# Minimal circuit search
# ===========================================================================
def _compute_node_positions(components, y_spacing=1.4):
    """Layout helper: x = layer index, y = spread across same-layer nodes."""
    labels = [f"L{d['layer']}-{d['component']}" for d in components]
    layer_of = {f"L{d['layer']}-{d['component']}": d["layer"] for d in components}
    layer_groups = {}
    for lbl in labels:
        layer_groups.setdefault(layer_of[lbl], []).append(lbl)
    pos = {}
    for l, names in sorted(layer_groups.items()):
        names_sorted = sorted(names)
        for i, n in enumerate(names_sorted):
            pos[n] = (l, (i - (len(names_sorted) - 1) / 2) * y_spacing)
    return pos, layer_of


def plot_combination_pathways(all_components, combinations_to_show, role_map=None, save_dir=GRAPH_DIR,
                               filename="minimal_circuits.png", title_suffix="", n_cols=4):
    """
    Faceted grid, one panel per tested combination, with an X marking each
    ablated component on top of its role-colored marker. A faceted layout
    avoids the overlap that connecting lines would cause once several
    combinations, especially nested ones, are drawn together.
    """
    pos, layer_of = _compute_node_positions(all_components)
    role_colors = {"detector": "orange", "aggregator": "crimson", "amplifier": "purple", "bystander": "grey"}
    n = len(combinations_to_show)
    if n == 0:
        return
    n_cols_eff = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols_eff))
    layers_present = sorted(set(layer_of.values()))
    all_y = [p[1] for p in pos.values()]
    fig, axes = plt.subplots(n_rows, n_cols_eff, figsize=(4.3 * n_cols_eff, 3.8 * n_rows), squeeze=False)

    for idx, combo in enumerate(combinations_to_show):
        ax = axes[idx // n_cols_eff][idx % n_cols_eff]
        member_set = set(combo["members"])
        for n_lbl, (x, y) in pos.items():
            role = role_map.get(n_lbl) if role_map else None
            ax.scatter([x], [y], s=260, color=role_colors.get(role, "lightgrey"), edgecolor="black", linewidth=0.8, zorder=3)
            if n_lbl in member_set:
                sz = 0.30
                ax.plot([x - sz, x + sz], [y - sz, y + sz], color="black", lw=1.7, zorder=5)
                ax.plot([x - sz, x + sz], [y + sz, y - sz], color="black", lw=1.7, zorder=5)
                ax.annotate(n_lbl, (x, y - 0.42), fontsize=5.5, ha="center", va="top", zorder=6)
        ax.set_xticks(layers_present); ax.set_xticklabels([f"L{l}" for l in layers_present], fontsize=6)
        ax.set_yticks([])
        ax.set_xlim(min(layers_present) - 0.6, max(layers_present) + 0.6)
        ax.set_ylim(min(all_y) - 0.7, max(all_y) + 0.6)
        ax.set_title(f"{combo['label']}\nRA={combo['ra']:.2f}, ASR={combo['asr']:.2f}", fontsize=8)
        ax.grid(True, alpha=0.15)
    for idx in range(n, n_rows * n_cols_eff):
        axes[idx // n_cols_eff][idx % n_cols_eff].axis("off")

    role_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=9,
                                markeredgecolor="black", label=r.capitalize()) for r, c in role_colors.items()]
    x_handle = plt.Line2D([0], [0], marker='x', color='black', markersize=9, lw=0, markeredgewidth=2, label="Ablated")
    fig.legend(handles=role_handles + [x_handle], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0 + 0.35 / n_rows), fontsize=8)
    fig.suptitle("Tested ablation combinations" + title_suffix, fontsize=12, fontweight="bold", y=1.0 + 0.55 / n_rows)
    plt.tight_layout(); plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight"); plt.close()


def greedy_minimal_circuit_search(model, loader, candidate_components, target_class, device, position_mode="cls",
                                   position=0, max_batches=None, ra_tolerance=0.02, save_dir=GRAPH_DIR):
    """
    Approximate minimal circuit search: starting from an empty set, greedily
    adds whichever remaining candidate improves joint RA the most, based on
    measured gain rather than a pre-sorted rank (rank cannot account for
    redundancy or synergy between components). O(K^2) evaluations for K
    candidates, tractable for pools too large for exhaustive search. Runs
    through every candidate rather than stopping early, so the full
    trajectory is visible.
    """
    candidates = [(d["layer"], d["component"]) for d in candidate_components]
    full_asr, full_ra = _joint_necessity_eval(model, loader, candidates, target_class, device, position_mode, position, max_batches)
    selected, remaining, history = [], list(candidates), []
    prev_ra = 0.0
    minimal_size = None
    while remaining:
        best_gain, best_c, best_ra, best_asr = -1, None, None, None
        for c in remaining:
            asr, ra = _joint_necessity_eval(model, loader, selected + [c], target_class, device, position_mode, position, max_batches)
            if ra - prev_ra > best_gain:
                best_gain, best_c, best_ra, best_asr = ra - prev_ra, c, ra, asr
        selected.append(best_c)
        remaining.remove(best_c)
        prev_ra = best_ra
        cumulative_set = [f"L{l}-{c}" for (l, c) in selected]
        history.append({"added": f"L{best_c[0]}-{best_c[1]}", "set_size": len(selected),
                         "cumulative_set": cumulative_set, "asr": best_asr, "ra": best_ra})
        if minimal_size is None and best_ra >= full_ra - ra_tolerance:
            minimal_size = len(selected)

    _plot_greedy_minimal_circuit(history, full_ra, minimal_size, position_mode, save_dir)
    return {"selected_components_in_order": [h["added"] for h in history], "history": history,
            "minimal_set_size": minimal_size, "full_set_asr": full_asr, "full_set_ra": full_ra, "position_mode": position_mode}


def _plot_greedy_minimal_circuit(history, full_ra, minimal_size, position_mode, save_dir=GRAPH_DIR):
    set_sizes = [h["set_size"] for h in history]
    ra_vals = [h["ra"] for h in history]
    asr_vals = [h["asr"] for h in history]
    fig, ax = plt.subplots(figsize=(max(9, len(history) * 1.3), 6.5))
    ax.plot(set_sizes, ra_vals, "go-", lw=2, markersize=7, label="RA")
    ax.plot(set_sizes, asr_vals, "ro-", lw=2, markersize=7, label="ASR")
    ax.axhline(full_ra, color="grey", ls="--", alpha=0.6, label=f"Full set RA = {full_ra:.3f}")
    if minimal_size is not None:
        idx = minimal_size - 1
        ax.axvspan(minimal_size - 0.5, max(set_sizes) + 0.5, color="darkorange", alpha=0.06, zorder=0)
        ax.scatter([set_sizes[idx]], [ra_vals[idx]], marker="D", s=130, color="darkorange", edgecolor="black",
                   zorder=6, label=f"Minimal set: size {minimal_size}")
    for x, h in zip(set_sizes, history):
        ax.annotate(f"+{h['added']}", (x, -0.14), rotation=45, fontsize=7, ha="right", va="top", annotation_clip=False)
    scope = "CLS position only" if position_mode == "cls" else "every position"
    ax.set_xlabel("Components jointly ablated so far (cumulative, greedy order)", labelpad=55)
    ax.set_ylabel("Rate"); ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(set_sizes); ax.set_xticklabels([])
    ax.set_title(f"Greedy minimal circuit search, {scope} (approximate)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_dir / f"minimal_circuit_greedy_{position_mode}.png", dpi=150, bbox_inches="tight"); plt.close()


def exhaustive_minimal_circuit_search(model, loader, candidate_components, target_class, device, position_mode="cls",
                                       max_batches=None, ra_tolerance=0.02, max_candidates=14, save_dir=GRAPH_DIR):
    """
    True minimal circuit search: tests every combination of candidates, in
    order of increasing size, without stopping early once tolerance is
    reached, so larger sizes are also checked for anything better. Only
    practical for small pools (max_candidates=14 default, 16383 combinations).
    """
    candidates = [(d["layer"], d["component"]) for d in candidate_components]
    if len(candidates) > max_candidates:
        raise ValueError(f"{len(candidates)} candidates exceeds max_candidates={max_candidates}, "
                          f"reduce the pool or use greedy_minimal_circuit_search instead.")
    full_asr, full_ra = _joint_necessity_eval(model, loader, candidates, target_class, device, position_mode, max_batches=max_batches)
    best_by_size = {}
    minimal_size = None
    for size in range(1, len(candidates) + 1):
        best_ra, best_combo, best_asr = -1, None, None
        for combo in combinations(candidates, size):
            asr, ra = _joint_necessity_eval(model, loader, list(combo), target_class, device, position_mode, max_batches=max_batches)
            if ra > best_ra:
                best_ra, best_combo, best_asr = ra, combo, asr
        best_by_size[size] = {"combo": [f"L{l}-{c}" for (l, c) in best_combo], "ra": best_ra, "asr": best_asr}
        if minimal_size is None and best_ra >= full_ra - ra_tolerance:
            minimal_size = size

    _plot_exhaustive_minimal_circuit(best_by_size, full_ra, minimal_size, position_mode, save_dir)
    return {"best_by_size": best_by_size, "full_set_asr": full_asr, "full_set_ra": full_ra,
            "minimal_size": minimal_size, "position_mode": position_mode}


def _plot_exhaustive_minimal_circuit(best_by_size, full_ra, minimal_size, position_mode, save_dir=GRAPH_DIR):
    sizes = sorted(best_by_size.keys())
    ra_vals = [best_by_size[s]["ra"] for s in sizes]
    asr_vals = [best_by_size[s]["asr"] for s in sizes]
    fig, ax = plt.subplots(figsize=(max(9, len(sizes) * 1.3), 6.5))
    ax.plot(sizes, ra_vals, "go-", lw=2, markersize=7, label="RA")
    ax.plot(sizes, asr_vals, "ro-", lw=2, markersize=7, label="ASR")
    ax.axhline(full_ra, color="grey", ls="--", alpha=0.6, label=f"Full set RA = {full_ra:.3f}")
    if minimal_size is not None:
        idx = sizes.index(minimal_size)
        ax.axvspan(minimal_size - 0.5, max(sizes) + 0.5, color="darkorange", alpha=0.06, zorder=0)
        ax.scatter([sizes[idx]], [ra_vals[idx]], marker="D", s=130, color="darkorange", edgecolor="black",
                   zorder=6, label=f"True minimum: size {minimal_size}")
    for x, s in zip(sizes, sizes):
        ax.annotate(", ".join(best_by_size[s]["combo"]), (x, -0.14), rotation=45, fontsize=6.5, ha="right", va="top", annotation_clip=False)
    scope = "CLS position only" if position_mode == "cls" else "every position"
    ax.set_xlabel("Combination size (best combination of exactly this many components)", labelpad=75)
    ax.set_ylabel("Rate"); ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(sizes); ax.set_xticklabels([])
    ax.set_title(f"Exhaustive minimal circuit search, {scope} (true minimum)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_dir / f"minimal_circuit_exhaustive_{position_mode}.png", dpi=150, bbox_inches="tight"); plt.close()


def plot_all_minimal_circuits(candidate_components, greedy_result, exhaustive_result, role_map=None, save_dir=GRAPH_DIR):
    """Merges greedy and exhaustive search results onto one faceted grid, deduplicated by member set."""
    combos, seen = [], set()

    def _add(label, members, ra, asr):
        key = frozenset(members)
        if key not in seen:
            seen.add(key)
            combos.append({"label": label, "members": members, "ra": ra, "asr": asr})

    if greedy_result is not None:
        for h in greedy_result["history"]:
            _add(f"greedy size {h['set_size']}", h["cumulative_set"], h["ra"], h["asr"])
    if exhaustive_result is not None:
        for size, info in sorted(exhaustive_result["best_by_size"].items()):
            _add(f"exhaustive size {size}", info["combo"], info["ra"], info["asr"])
    if not combos:
        return
    mode = (greedy_result or exhaustive_result or {}).get("position_mode", "cls")
    plot_combination_pathways(candidate_components, combos, role_map, save_dir,
                               filename=f"minimal_circuits_combined_{mode}.png", title_suffix=f", mode={mode}")


def _plot_activation_patching_heatmap(attribution, component_list, positions, save_dir, filename, shared_abs_max=None, title_suffix=""):
    """Heatmap of components (rows) by token positions (columns). shared_abs_max lets two heatmaps use the same color scale for honest comparison."""
    mat = np.array([[attribution[(l, c, p)] for p in positions] for (l, c) in component_list])
    labels_y = [f"L{l}-{c}" for (l, c) in component_list]
    fig, ax = plt.subplots(figsize=(min(0.28 * len(positions) + 3, 22), 0.45 * len(component_list) + 2.2))
    this_abs_max = np.abs(mat).max() + 1e-8
    abs_max = shared_abs_max if shared_abs_max is not None else this_abs_max
    norm = SymLogNorm(linthresh=max(abs_max * 0.01, 1e-3), vmin=-abs_max, vmax=abs_max)
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", norm=norm)
    tick_stride = max(1, len(positions) // 15)
    ax.set_xticks(range(0, len(positions), tick_stride))
    ax.set_xticklabels([positions[i] for i in range(0, len(positions), tick_stride)])
    ax.set_xlabel("Token position"); ax.set_yticks(range(len(component_list))); ax.set_yticklabels(labels_y)
    ax.set_ylabel("Component")
    if len(positions) <= 20:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="Delta target-class logit")
    ax.set_title("Activation patching: sufficiency by component and token position" + title_suffix)
    plt.tight_layout(); plt.savefig(save_dir / filename, dpi=150); plt.close()
    return this_abs_max


# ===========================================================================
# Path patching
# ===========================================================================
def _make_freeze_attn_hook(clean_head_cache_l, override_head=None, override_value=None):
    """Rebuilds attention output purely from cached clean per-head values, except override_head if given."""
    def hook(module, inp, out):
        heads = clean_head_cache_l.clone()
        if override_head is not None:
            heads[:, override_head, :, :] = override_value
        summed = heads.sum(dim=1)
        if module.out_proj.bias is not None:
            summed = summed + module.out_proj.bias
        return (summed, out[1] if isinstance(out, tuple) else None)
    return hook


def _make_freeze_mlp_hook(clean_mlp_cache_l, override_value=None):
    def hook(module, inp, out):
        return override_value if override_value is not None else clean_mlp_cache_l
    return hook


def _make_capture_hook(store, key):
    """Records a component's naturally computed output without altering it, used to read the receiver in path patching."""
    def hook(module, inp, out):
        store[key] = out[0].detach() if isinstance(out, tuple) else out.detach()
        return out
    return hook


def _freeze_all_except(model, clean_head_cache, clean_mlp_cache, override_layer=None, override_comp=None,
                        override_value=None, skip_layer=None, skip_component=None, capture_store=None, capture_key=None):
    """Freezes every component to its clean value except override (set to override_value) and skip (left to react naturally)."""
    hooks = []
    for l, blk in enumerate(model.transformer):
        skip_attn = l == skip_layer and skip_component == "attn"
        skip_mlp = l == skip_layer and skip_component == "mlp"
        override_head = override_head_val = override_mlp_val = None
        if l == override_layer:
            if override_comp.startswith("head_"):
                override_head, override_head_val = int(override_comp.split("_")[1]), override_value
            elif override_comp == "mlp":
                override_mlp_val = override_value

        if skip_attn:
            if capture_store is not None:
                hooks.append(blk.attn.register_forward_hook(_make_capture_hook(capture_store, capture_key)))
        else:
            hooks.append(blk.attn.register_forward_hook(_make_freeze_attn_hook(clean_head_cache[l], override_head, override_head_val)))
        if skip_mlp:
            if capture_store is not None:
                hooks.append(blk.mlp.register_forward_hook(_make_capture_hook(capture_store, capture_key)))
        else:
            hooks.append(blk.mlp.register_forward_hook(_make_freeze_mlp_hook(clean_mlp_cache[l], override_mlp_val)))
    return hooks


def path_patch_component_pair(model, clean_head_cache, clean_mlp_cache, triggered_head_cache, triggered_mlp_cache,
                               layer_A, comp_A, layer_B, comp_B, img_c, target_class, device, position_mode="all", position=0):
    """
    Standard two-run path patching: isolates the direct A -> B edge by
    freezing every other component to clean, corrupting only A (at one
    position for "cls" mode, every position for "all"), letting B react
    naturally, then forcing that reaction into a second clean run to read
    the resulting target-class logit shift.
    """
    if comp_A.startswith("head_"):
        h_idx = int(comp_A.split("_")[1])
        full_triggered, full_clean = triggered_head_cache[layer_A][:, h_idx], clean_head_cache[layer_A][:, h_idx]
    else:
        full_triggered, full_clean = triggered_mlp_cache[layer_A], clean_mlp_cache[layer_A]

    if position_mode == "cls":
        override_val = full_clean.clone()
        override_val[:, position, :] = full_triggered[:, position, :]
    else:
        override_val = full_triggered

    skip_component = "attn" if comp_B.startswith("head_") else "mlp"
    captured = {}
    hooks = _freeze_all_except(model, clean_head_cache, clean_mlp_cache, layer_A, comp_A, override_val,
                                layer_B, skip_component, captured, "b_natural")
    with torch.no_grad():
        model(img_c)
    for h in hooks:
        h.remove()

    if comp_B.startswith("head_"):
        b_head_idx = int(comp_B.split("_")[1])
        ln_input_at_B = {}

        def _cap_ln(mod, inp, out):
            ln_input_at_B["x"] = inp[0].detach()

        hook2 = model.transformer[layer_B].attn.register_forward_hook(_cap_ln)
        hooks2 = _freeze_all_except(model, clean_head_cache, clean_mlp_cache, layer_A, comp_A, override_val)
        with torch.no_grad():
            model(img_c)
        hook2.remove()
        for h in hooks2:
            h.remove()
        head_contribs_B, _ = _manual_per_head_attention(model.transformer[layer_B].attn, ln_input_at_B["x"])
        b_under_patch = head_contribs_B[:, b_head_idx]
    else:
        b_under_patch = captured["b_natural"]

    hooks3 = _freeze_all_except(model, clean_head_cache, clean_mlp_cache, layer_B, comp_B, b_under_patch)
    with torch.no_grad():
        patched_logits = model(img_c)
    for h in hooks3:
        h.remove()
    with torch.no_grad():
        baseline_hooks = _freeze_all_except(model, clean_head_cache, clean_mlp_cache)
        baseline_logits = model(img_c)
        for h in baseline_hooks:
            h.remove()
    return (patched_logits[:, target_class] - baseline_logits[:, target_class]).mean().item()


def _valid_path_patching_pair(a, b, max_layer_gap=None):
    """B must be at the same or a later layer than A; within a layer only attn -> mlp is valid; max_layer_gap caps distance."""
    la, lb = a["layer"], b["layer"]
    if lb < la:
        return False
    if lb == la and not (a["component"].startswith("head_") and b["component"] == "mlp"):
        return False
    if max_layer_gap is not None and (lb - la) > max_layer_gap:
        return False
    return True


def run_path_patching(model, loader, top_components, target_class, device, max_pairs=200, max_batches=1,
                       max_layer_gap=None, position_modes=("cls", "all"), role_map=None, save_dir=GRAPH_DIR):
    """Sweeps every valid ordered pair among top_components, in both CLS-only and all-position sender modes. Returns {mode: edge_scores}."""
    pairs_all = [(a, b) for a, b in product(top_components, top_components) if _valid_path_patching_pair(a, b, max_layer_gap)]
    pairs = pairs_all[:max_pairs]
    all_edge_scores = {}
    for mode in position_modes:
        delta_accum = {}
        for b_idx, (img_c, _, img_b, _) in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
            img_c, img_b = img_c.to(device), img_b.to(device)
            clean_head_cache, clean_mlp_cache = _cache_batch_components(model, img_c, device)
            trig_head_cache, trig_mlp_cache = _cache_batch_components(model, img_b, device)
            for a, b in tqdm(pairs, desc=f"Path patching, mode={mode}"):
                delta = path_patch_component_pair(model, clean_head_cache, clean_mlp_cache, trig_head_cache, trig_mlp_cache,
                                                    a["layer"], a["component"], b["layer"], b["component"], img_c, target_class, device, mode)
                delta_accum.setdefault((a["layer"], a["component"], b["layer"], b["component"]), []).append(delta)

        edge_scores = [{"A": f"L{a['layer']}-{a['component']}", "B": f"L{b['layer']}-{b['component']}",
                        "layer_A": a["layer"], "layer_B": b["layer"],
                        "delta": float(np.mean(delta_accum[(a["layer"], a["component"], b["layer"], b["component"])]))}
                       for a, b in pairs]
        scope = "sender at CLS only" if mode == "cls" else "sender at every position"
        _plot_path_patching_graph(edge_scores, role_map, save_dir, f"path_patching_graph_{mode}.png", f"Path patching, {scope}")
        _plot_path_patching_edge_ranking(edge_scores, save_dir, f"path_patching_ranking_{mode}.png")
        all_edge_scores[mode] = edge_scores
    return all_edge_scores


def run_full_depth_path_patching(model, loader, all_scored, target_class, device, max_batches=1,
                                  position_mode="all", role_map=None, save_dir=GRAPH_DIR):
    """Complete circuit: every component at every layer, adjacent-layer edges only, since full pairwise coverage is combinatorially infeasible at this scale."""
    per_layer_components = select_top_component_per_layer(all_scored, top_per_layer=None)
    pairs = [(a, b) for a, b in product(per_layer_components, per_layer_components) if _valid_path_patching_pair(a, b, max_layer_gap=1)]
    delta_accum = {}
    for b_idx, (img_c, _, img_b, _) in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        clean_head_cache, clean_mlp_cache = _cache_batch_components(model, img_c, device)
        trig_head_cache, trig_mlp_cache = _cache_batch_components(model, img_b, device)
        for a, b in tqdm(pairs, desc="Full-depth path patching"):
            delta = path_patch_component_pair(model, clean_head_cache, clean_mlp_cache, trig_head_cache, trig_mlp_cache,
                                               a["layer"], a["component"], b["layer"], b["component"], img_c, target_class, device, position_mode)
            delta_accum.setdefault((a["layer"], a["component"], b["layer"], b["component"]), []).append(delta)

    edge_scores = [{"A": f"L{a['layer']}-{a['component']}", "B": f"L{b['layer']}-{b['component']}",
                    "layer_A": a["layer"], "layer_B": b["layer"],
                    "delta": float(np.mean(delta_accum[(a["layer"], a["component"], b["layer"], b["component"])]))}
                   for a, b in pairs]
    _plot_path_patching_graph(edge_scores, role_map, save_dir, "path_patching_full_depth.png",
                               "Full-depth circuit, adjacent-layer edges")
    return edge_scores, per_layer_components


def _format_delta(delta):
    return f"{delta:.1e}" if delta != 0 and abs(delta) < 0.01 else f"{delta:.2f}"


def _plot_path_patching_graph(edge_scores, role_map=None, save_dir=GRAPH_DIR, filename="path_patching_graph.png",
                               title="Path patching", highlight_nodes=None):
    """Directed circuit diagram: nodes positioned by layer, colored by functional role, edges show path-patched A -> B effects."""
    nodes = sorted(set([e["A"] for e in edge_scores] + [e["B"] for e in edge_scores]))
    layer_of = {}
    for e in edge_scores:
        layer_of[e["A"]], layer_of[e["B"]] = e["layer_A"], e["layer_B"]
    layer_groups = {}
    for n in nodes:
        layer_groups.setdefault(layer_of[n], []).append(n)
    pos = {}
    for l, names in sorted(layer_groups.items()):
        for i, n in enumerate(sorted(names)):
            pos[n] = (l, (i - (len(names) - 1) / 2) * 1.4)

    role_colors = {"detector": "orange", "aggregator": "crimson", "amplifier": "purple", "bystander": "grey"}
    highlight_set = set(highlight_nodes) if highlight_nodes is not None else None
    n_layers_present = len(set(layer_of.values()))
    max_nodes = max(len(v) for v in layer_groups.values())
    fig, ax = plt.subplots(figsize=(max(12, n_layers_present * 2.2), max(6, max_nodes * 1.9)))
    max_abs = max(abs(e["delta"]) for e in edge_scores) + 1e-8

    for e in edge_scores:
        x1, y1 = pos[e["A"]]; x2, y2 = pos[e["B"]]
        strength = abs(e["delta"]) / max_abs
        is_pos = e["delta"] > 0
        color = "crimson" if is_pos else "steelblue"
        edge_in_circuit = highlight_set is not None and e["A"] in highlight_set and e["B"] in highlight_set
        if highlight_set is not None and not edge_in_circuit:
            lw, alpha, color, z = 0.5, 0.08, "lightgrey", 1
        else:
            lw, alpha, z = 0.8 + 4.5 * strength, 0.35 + 0.55 * strength, 3
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha, linestyle="-" if is_pos else "--", shrinkA=22, shrinkB=22), zorder=z)
        if highlight_set is None or edge_in_circuit:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, _format_delta(e['delta']), fontsize=6.5, color=color, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.5, alpha=0.9), zorder=7)

    for n, (x, y) in pos.items():
        role = role_map.get(n) if role_map else None
        in_circuit = highlight_set is None or n in highlight_set
        node_color = role_colors.get(role, "lightgrey") if in_circuit else "whitesmoke"
        ax.scatter([x], [y], s=750 if in_circuit else 400, color=node_color, edgecolor="black" if in_circuit else "lightgrey",
                   linewidth=1.3 if in_circuit else 0.8, zorder=6 if in_circuit else 4)
        ax.annotate(n, (x, y - 0.34), fontsize=8, fontweight="bold" if in_circuit else "normal", ha="center", va="top", zorder=6,
                    alpha=1.0 if in_circuit else 0.35, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", lw=0.5, alpha=0.9))

    layers_present = sorted(set(layer_of.values()))
    ax.set_xticks(layers_present); ax.set_xticklabels([f"L{l}" for l in layers_present], fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(min(layers_present) - 0.7, max(layers_present) + 0.7)
    all_y = [p[1] for p in pos.values()]
    ax.set_ylim(min(all_y) - 1.0, max(all_y) + 0.7)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.15)

    roles_present = set(role_map.values()) if role_map else set()
    role_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=role_colors[r], markersize=15,
                                markeredgecolor="black", label=r.capitalize()) for r in role_colors if r in roles_present]
    if role_handles:
        legend1 = ax.legend(handles=role_handles, title="Functional role", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
        ax.add_artist(legend1)
    edge_handles = [plt.Line2D([0], [0], color="crimson", lw=3, label="Promotes target class"),
                     plt.Line2D([0], [0], color="steelblue", lw=3, ls="--", label="Suppresses target class")]
    ax.legend(handles=edge_handles, title="Edge direction", loc="upper left", bbox_to_anchor=(1.01, 0.5 if role_handles else 1.0), fontsize=9)
    plt.tight_layout(); plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight"); plt.close()


def _plot_path_patching_edge_ranking(edge_scores, save_dir=GRAPH_DIR, filename="path_patching_ranking.png"):
    sorted_edges = sorted(edge_scores, key=lambda e: abs(e["delta"]), reverse=True)
    labels = [f"{e['A']} -> {e['B']}" for e in sorted_edges]
    vals = [e["delta"] for e in sorted_edges]
    colors = ["crimson" if v > 0 else "steelblue" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(sorted_edges) + 2))
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor="black")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Delta target-class logit")
    ax.set_title("Path patching edges ranked by effect size")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(save_dir / filename, dpi=150); plt.close()


# ===========================================================================
# Layer-wise activation patching
# ===========================================================================
def layer_wise_activation_patching(model, loader, target_class, device, max_batches=None, save_dir=GRAPH_DIR):
    """
    Layer-level sufficiency test: patches only layer l's own write onto an
    otherwise-clean run, not the whole downstream state. Overwriting the
    whole cumulative state instead would make every layer's delta collapse
    to the same number, since it just replays the true triggered
    continuation regardless of where you patch. Isolating the own write
    mirrors how component-level activation patching isolates one head's
    own write rather than swapping the whole downstream state.
    """
    n_layers = len(model.transformer)
    results = {l: [] for l in range(n_layers)}
    for b_idx, (img_c, _, img_b, _) in enumerate(tqdm(loader, desc="Layer-wise activation patching")):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        triggered_inputs, triggered_outputs = {}, {}

        def _cap(mod, inp, out, idx):
            triggered_inputs[idx] = inp[0].detach()
            triggered_outputs[idx] = out.detach()

        hooks = [blk.register_forward_hook(lambda m, i, o, idx=i_: _cap(m, i, o, idx)) for i_, blk in enumerate(model.transformer)]
        with torch.no_grad():
            model(img_b)
        for h in hooks:
            h.remove()
        triggered_own_write = {l: triggered_outputs[l] - triggered_inputs[l] for l in range(n_layers)}

        with torch.no_grad():
            baseline_target = model(img_c)[:, target_class]
            for l in range(n_layers):
                def _patch(mod, inp, out, write=triggered_own_write[l]):
                    return inp[0] + write
                hook = model.transformer[l].register_forward_hook(_patch)
                delta = (model(img_c)[:, target_class] - baseline_target).mean().item()
                hook.remove()
                results[l].append(delta)

    attribution = {l: float(np.mean(v)) for l, v in results.items()}
    _plot_layer_wise_activation_patching(attribution, save_dir)
    return attribution


def compare_layer_sufficiency_vs_component_sum(layer_wise_attribution, activation_patching_attribution, top_components,
                                                cls_position=0, save_dir=GRAPH_DIR):
    """
    Compares each layer's whole causal sufficiency against the sum of its
    own top-K components' individual sufficiency. A small gap supports the
    premise that a sparse subset explains most of the effect, the premise
    a component-level defense needs to hold.
    """
    by_layer = {}
    for d in top_components:
        by_layer.setdefault(d["layer"], []).append(d["component"])
    layers, whole_vals, sum_vals = [], [], []
    for l, comps in sorted(by_layer.items()):
        if l not in layer_wise_attribution:
            continue
        whole = layer_wise_attribution[l]
        comp_sum = sum(activation_patching_attribution.get((l, c, cls_position), 0.0) for c in comps)
        layers.append(l); whole_vals.append(whole); sum_vals.append(comp_sum)

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 1.1), 5))
    x = np.arange(len(layers)); width = 0.35
    ax.bar(x - width / 2, whole_vals, width, label="Whole layer", color="#1f77b4")
    ax.bar(x + width / 2, sum_vals, width, label="Sum of top-K components", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Delta target-class logit")
    ax.set_title("Whole-layer sufficiency vs. sum of its top-K components")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(save_dir / "layer_vs_topk_sufficiency.png", dpi=150); plt.close()
    return {"layers": layers, "whole_layer": whole_vals, "component_sum": sum_vals}


def _plot_layer_wise_activation_patching(attribution, save_dir=GRAPH_DIR):
    layers = sorted(attribution.keys())
    vals = [attribution[l] for l in layers]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(l) for l in layers], vals, color=["crimson" if v > 0 else "steelblue" for v in vals])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Delta target-class logit")
    ax.set_title("Layer-wise activation patching: sufficiency of each layer's own write")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(save_dir / "layer_wise_activation_patching.png", dpi=150); plt.close()


# ===========================================================================
# Functional role characterization
# ===========================================================================
@torch.no_grad()
def compute_attention_to_trigger(model, loader, device, trigger_position=None, patch_size=4, cls_offset=1, max_batches=None):
    """Mean attention weight directed toward the trigger position(s), per head and layer, on triggered inputs."""
    n_layers = len(model.transformer)
    ln_inputs = {}

    def _hook(mod, inp, out, idx):
        ln_inputs[idx] = inp[0].detach()

    hooks = [blk.attn.register_forward_hook(lambda m, i, o, idx=i_: _hook(m, i, o, idx)) for i_, blk in enumerate(model.transformer)]
    accum = {l: [] for l in range(n_layers)}
    for b_idx, (img_c, _, img_b, _) in enumerate(tqdm(loader, desc="Attention to trigger")):
        if max_batches and b_idx >= max_batches:
            break
        if trigger_position is None:
            trig_positions_per_sample = detect_trigger_patch_positions(img_c, img_b, patch_size, cls_offset)
        model(img_b.to(device))
        for l in range(n_layers):
            _, attn_weights = _manual_per_head_attention(model.transformer[l].attn, ln_inputs[l])
            aw = attn_weights.mean(dim=2).cpu()
            if trigger_position is not None:
                accum[l].append(aw[:, :, trigger_position])
            else:
                accum[l].append(torch.stack([aw[i, :, trig_positions_per_sample[i]].mean(dim=-1) for i in range(aw.shape[0])]))
    for h in hooks:
        h.remove()
    return {l: torch.cat(v, dim=0).mean(0) for l, v in accum.items()}


def classify_functional_roles(attn_to_trigger, activation_patching_results, top_components, cls_position=0,
                               full_depth_attribution=None, save_dir=GRAPH_DIR):
    """
    Classifies each top-K component into detector (heads only, high
    attention to trigger), aggregator (high attention and high patch
    effect), amplifier (high patch effect, low or no attention), or
    bystander (low on both). Thresholds are computed from the global
    population (every head in the model for attention, every component if
    full_depth_attribution is given for patch effect) rather than only
    top_components, since a median split within an already-selected elite
    subset would always label roughly half as "high" regardless of whether
    they are high in absolute terms.
    """
    records = []
    for d in top_components:
        l, comp = d["layer"], d["component"]
        patch_score = activation_patching_results.get((l, comp, cls_position), 0.0)
        attn_score = attn_to_trigger[l][int(comp.split("_")[1])].item() if comp.startswith("head_") else None
        records.append({"layer": l, "component": comp, "label": f"L{l}-{comp}", "attn_to_trigger": attn_score, "patch_effect": patch_score})

    all_head_attn = np.array([v.item() for l in attn_to_trigger for v in attn_to_trigger[l]])
    attn_thresh = np.median(all_head_attn) if len(all_head_attn) > 0 else None
    if full_depth_attribution is not None:
        patch_abs = np.array([abs(v) for v in full_depth_attribution.values()])
    else:
        patch_abs = np.array([abs(r["patch_effect"]) for r in records])
        print("Warning: no full_depth_attribution given, patch-effect threshold is relative to top_components only.")
    patch_thresh = np.median(patch_abs) if len(patch_abs) else 0.0

    for r in records:
        high_patch = abs(r["patch_effect"]) >= patch_thresh
        if r["attn_to_trigger"] is not None:
            high_attn = r["attn_to_trigger"] >= attn_thresh
            r["role"] = "aggregator" if high_attn and high_patch else "detector" if high_attn else "amplifier" if high_patch else "bystander"
        else:
            r["role"] = "amplifier" if high_patch else "bystander"

    role_colors = {"detector": "orange", "aggregator": "crimson", "amplifier": "purple", "bystander": "grey"}
    records_sorted = sorted(records, key=lambda r: abs(r["patch_effect"]))
    labels = [r["label"] for r in records_sorted]
    vals = [r["patch_effect"] for r in records_sorted]
    colors = [role_colors[r["role"]] for r in records_sorted]

    fig, ax = plt.subplots(figsize=(11.5, 0.55 * len(records_sorted) + 2.5))
    ax.barh(labels, vals, color=colors, edgecolor="black")
    max_abs_val = max(abs(v) for v in vals) + 1e-8
    text_x = max(max(v, 0) for v in vals) + 0.06 * max_abs_val
    for i, r in enumerate(records_sorted):
        attn_str = f"{r['attn_to_trigger']:.3f}" if r["attn_to_trigger"] is not None else "n/a (MLP)"
        ax.text(text_x, i, f"patch effect = {r['patch_effect']:.3f}   attn to trigger = {attn_str}", va="center", ha="left", fontsize=7.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(min(min(vals, default=0), 0) - 0.05 * max_abs_val, text_x + 0.55 * max_abs_val)
    ax.set_xlabel("Activation patching Delta at CLS position")
    ax.set_title("Functional roles of identified circuit components")
    roles_present = [r for r in role_colors if r in set(rec["role"] for rec in records)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=role_colors[r], ec="black") for r in roles_present]
    ax.legend(handles, [r.capitalize() for r in roles_present], title="Role", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(save_dir / "functional_roles.png", dpi=150, bbox_inches="tight"); plt.close()
    return records


# ===========================================================================
# Steering magnitude sweeps
# ===========================================================================
def component_activation_steering(model, loader, layer_idx, component, direction, mode, target_class, device,
                                   alpha=1.0, max_batches=None, steer_all_tokens=False):
    """Adds or subtracts alpha times direction to/from one component's own output, a steering intervention rather than a value swap."""
    d = direction.to(device)
    sign = 1 if mode == "positive" else -1

    def hook(mod, inp, out):
        if component.startswith("head_"):
            h = int(component.split("_")[1])
            head_contribs, _ = _manual_per_head_attention(mod, inp[0])
            head_contribs = head_contribs.clone()
            if steer_all_tokens:
                head_contribs[:, h, :, :] += sign * alpha * d.unsqueeze(0)
            else:
                head_contribs[:, h, 0, :] += sign * alpha * d
            patched = head_contribs.sum(dim=1)
            if mod.out_proj.bias is not None:
                patched = patched + mod.out_proj.bias
            return (patched, out[1] if isinstance(out, tuple) else None)
        patched = out.clone()
        if steer_all_tokens:
            patched += sign * alpha * d.unsqueeze(0)
        else:
            patched[:, 0, :] += sign * alpha * d
        return patched

    blk = model.transformer[layer_idx]
    target_module = blk.attn if component.startswith("head_") else blk.mlp
    hook_handle = target_module.register_forward_hook(hook)
    correct = total = 0
    with torch.no_grad():
        for b_idx, (img_c, lbl_c, img_b, _) in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
            inputs = img_c.to(device) if mode == "positive" else img_b.to(device)
            preds = model(inputs).argmax(1).cpu()
            correct += (preds == target_class).sum().item() if mode == "positive" else (preds == lbl_c).sum().item()
            total += len(img_c)
    hook_handle.remove()
    return correct / total if total else 0.0


def sweep_multiple_components_steering_magnitude(model, loader, top_components, head_dirs, mlp_dirs,
                                                   layer_reference_direction_cls, layer_reference_idx, target_class, device,
                                                   n_components=4, alphas=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
                                                   layer_reference_direction_all=None, max_batches=None, save_dir=GRAPH_DIR):
    """Compares the top-N ranked components' own steering curves against the whole reference layer's curve, CLS-only and all-tokens."""
    if layer_reference_direction_all is None:
        layer_reference_direction_all = head_dirs[layer_reference_idx].sum(0) + mlp_dirs[layer_reference_idx]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_components, 1) + 1))

    def _layer_reference_curve(direction):
        if direction is None:
            return None, None
        asr, ra = [], []
        all_tok = direction.dim() == 2
        for alpha in alphas:
            for mode, store in [("positive", asr), ("negative", ra)]:
                def _hook(mod, inp, out, a=alpha, s=1 if mode == "positive" else -1):
                    d = direction.to(device)
                    steered = out.clone()
                    steered += s * a * (d.unsqueeze(0) if all_tok else 0)
                    if not all_tok:
                        steered[:, 0, :] += s * a * d
                    return steered
                hook = model.transformer[layer_reference_idx].register_forward_hook(_hook)
                correct = total = 0
                with torch.no_grad():
                    for b_idx, (img_c, lbl_c, img_b, _) in enumerate(loader):
                        if max_batches and b_idx >= max_batches:
                            break
                        inputs = img_c.to(device) if mode == "positive" else img_b.to(device)
                        preds = model(inputs).argmax(1).cpu()
                        correct += (preds == target_class).sum().item() if mode == "positive" else (preds == lbl_c).sum().item()
                        total += len(img_c)
                hook.remove()
                store.append(correct / total if total else 0.0)
        return asr, ra

    layer_asr_cls, layer_ra_cls = _layer_reference_curve(layer_reference_direction_cls)
    layer_asr_all, layer_ra_all = _layer_reference_curve(layer_reference_direction_all)
    results = {"cls": {"components": {}}, "all": {"components": {}}, "layer_asr_cls": layer_asr_cls,
               "layer_ra_cls": layer_ra_cls, "layer_asr_all": layer_asr_all, "layer_ra_all": layer_ra_all}

    for mode_key, steer_all in [("cls", False), ("all", True)]:
        ax_asr, ax_ra = (axes[0, 0], axes[1, 0]) if mode_key == "cls" else (axes[0, 1], axes[1, 1])
        layer_asr = layer_asr_cls if mode_key == "cls" else layer_asr_all
        layer_ra = layer_ra_cls if mode_key == "cls" else layer_ra_all
        if layer_asr is not None:
            ax_asr.plot(alphas, layer_asr, "k--", lw=2.5, marker="D", label=f"Layer {layer_reference_idx} (reference)")
            ax_ra.plot(alphas, layer_ra, "k--", lw=2.5, marker="D", label=f"Layer {layer_reference_idx} (reference)")
        for idx, comp_info in enumerate(top_components[:n_components]):
            l, comp = comp_info["layer"], comp_info["component"]
            if comp.startswith("head_"):
                h = int(comp.split("_")[1])
                direction = head_dirs[l][h] if steer_all else head_dirs[l][h, 0, :]
            else:
                direction = mlp_dirs[l] if steer_all else mlp_dirs[l][0, :]
            asr_pos = [component_activation_steering(model, loader, l, comp, direction, "positive", target_class, device, a, max_batches, steer_all) for a in alphas]
            ra_neg = [component_activation_steering(model, loader, l, comp, direction, "negative", target_class, device, a, max_batches, steer_all) for a in alphas]
            label = f"L{l}-{comp}"
            ax_asr.plot(alphas, asr_pos, "o-", color=colors[idx], label=label)
            ax_ra.plot(alphas, ra_neg, "s-", color=colors[idx], label=label)
            results[mode_key]["components"][label] = {"asr_pos": asr_pos, "ra_neg": ra_neg}

    axes[0, 0].set_title("ASR+ (CLS-only steering)"); axes[1, 0].set_title("RA- (CLS-only steering)")
    axes[0, 1].set_title("ASR+ (all-tokens steering)"); axes[1, 1].set_title("RA- (all-tokens steering)")
    for ax in axes.flat:
        ax.axvline(1.0, color="grey", ls=":", alpha=0.5, label="Natural magnitude")
        ax.set_xlabel("Steering magnitude"); ax.set_ylabel("Rate")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3)
    fig.suptitle("Steering magnitude: top components vs. whole layer", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_dir / "steering_magnitude_sweep.png", dpi=150); plt.close()
    return results


# ===========================================================================
# Layer vs component defense proxy
# ===========================================================================
def compare_layer_vs_component_defense_proxy(model, loader, target_class, device, r_hat, component_set,
                                              token_scope="all", max_batches=None, save_dir=GRAPH_DIR):
    """
    Activation-level preview of the Phase 2 weight orthogonalization
    comparison. Projects h' = h - (h . r_hat) r_hat, the activation-level
    equivalent of W_new = W - r_hat r_hat^T W, since matrix multiplication
    is associative: projecting a weight matrix's output is identical to
    projecting the activation directly.

    The whole-layer condition targets every writing matrix in the model,
    the patch-embedding Linear plus every transformer block, matching
    global_orthogonalize's actual scope exactly. Each block's own write is
    isolated (out minus the incoming state) before projecting, since real
    weight orthogonalization only modifies a layer's own writing matrices
    and never touches the residual state arriving from earlier layers.

    token_scope "all" projects at every token position for both conditions,
    since a weight matrix has no concept of a single token position. 
    token_scope "cls" projects only the CLS position, a diagnostic-only 
    ablation that decomposes how much of the all-token effect comes from 
    CLS specifically, not a candidate defense.

    component_set is the identified minimal circuit, the one deliberate
    difference between the two conditions. Still an activation hook, not a
    real weight edit.
    """
    r = (r_hat / (r_hat.norm() + 1e-8)).to(device)

    def _project_out(tensor, scope):
        if scope == "cls":
            out = tensor.clone()
            coeff = (tensor[:, 0, :] * r).sum(dim=-1, keepdim=True)
            out[:, 0, :] = tensor[:, 0, :] - coeff * r
            return out
        coeff = (tensor * r).sum(dim=-1, keepdim=True)
        return tensor - coeff * r

    def _se(p, n):
        return float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0

    def _eval(hook_specs):
        hooks = [target.register_forward_hook(fn) for target, fn in hook_specs]
        correct_c = correct_b = correct_ra = total = 0
        with torch.no_grad():
            for b_idx, (img_c, lbl_c, img_b, lbl_b) in enumerate(loader):
                if max_batches and b_idx >= max_batches:
                    break
                img_c, lbl_c, img_b = img_c.to(device), lbl_c.to(device), img_b.to(device)
                pred_c = model(img_c).argmax(1)
                pred_b = model(img_b).argmax(1)
                correct_c += (pred_c == lbl_c).sum().item()
                correct_b += (pred_b == target_class).sum().item()
                correct_ra += (pred_b == lbl_c).sum().item()
                total += len(img_c)
        for h in hooks:
            h.remove()
        if total == 0:
            return {"acc": 0.0, "asr": 0.0, "ra": 0.0, "acc_se": 0.0, "asr_se": 0.0, "ra_se": 0.0, "n": 0}
        acc, asr, ra = correct_c / total, correct_b / total, correct_ra / total
        return {"acc": acc, "asr": asr, "ra": ra, "acc_se": _se(acc, total), "asr_se": _se(asr, total), "ra_se": _se(ra, total), "n": total}

    baseline = _eval([])

    patch_embed_linear = [m for m in model.to_patch_embedding.modules() if isinstance(m, nn.Linear)][0]

    def _patch_embed_hook(mod, inp, out):
        return _project_out(out, "all")

    def _make_layer_hook():
        def hook(mod, inp, out):
            own_write = out - inp[0]
            return inp[0] + _project_out(own_write, token_scope)
        return hook

    layer_wide = _eval([(patch_embed_linear, _patch_embed_hook)] + [(blk, _make_layer_hook()) for blk in model.transformer])

    comp_hook_specs = []
    for (l, comp) in component_set:
        blk = model.transformer[l]
        if comp.startswith("head_"):
            h_idx = int(comp.split("_")[1])

            def _mk(h_idx=h_idx):
                def hook(mod, inp, out):
                    head_contribs, _ = _manual_per_head_attention(mod, inp[0])
                    head_contribs = head_contribs.clone()
                    head_contribs[:, h_idx, :, :] = _project_out(head_contribs[:, h_idx, :, :], token_scope)
                    patched = head_contribs.sum(dim=1)
                    if mod.out_proj.bias is not None:
                        patched = patched + mod.out_proj.bias
                    return (patched, out[1] if isinstance(out, tuple) else None)
                return hook
            comp_hook_specs.append((blk.attn, _mk()))
        else:
            comp_hook_specs.append((blk.mlp, lambda mod, inp, out: _project_out(out, token_scope)))
    component_level = _eval(comp_hook_specs)

    results = {"token_scope": token_scope, "baseline": baseline, "layer_wide": layer_wide, "component_level": component_level}
    print(f"Defense proxy ({token_scope}): baseline ACC={baseline['acc']:.3f} ASR={baseline['asr']:.3f} RA={baseline['ra']:.3f} | "
          f"layer-wide ACC={layer_wide['acc']:.3f} ASR={layer_wide['asr']:.3f} RA={layer_wide['ra']:.3f} | "
          f"component ACC={component_level['acc']:.3f} ASR={component_level['asr']:.3f} RA={component_level['ra']:.3f}")
    _plot_layer_vs_component_comparison(results, save_dir)
    return results


def _plot_layer_vs_component_comparison(results, save_dir=GRAPH_DIR):
    token_scope = results.get("token_scope", "all")
    conditions = ["baseline", "layer_wide", "component_level"]
    condition_labels = ["No intervention", "Whole-model\nr_hat projection", "Component-level\nr_hat projection"]
    metrics = ["acc", "asr", "ra"]
    colors = {"acc": "#1f77b4", "asr": "#d62728", "ra": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(conditions)); width = 0.25
    for i, m in enumerate(metrics):
        vals = [results[c][m] for c in conditions]
        errs = [results[c][f"{m}_se"] for c in conditions]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=3, label=m.upper(), color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels(condition_labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rate")
    scope_label = "all tokens" if token_scope == "all" else "CLS only, diagnostic"
    ax.set_title(f"Defense proxy: whole-model vs. component-level r_hat projection ({scope_label})")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(save_dir / f"defense_proxy_{token_scope}.png", dpi=150); plt.close()


# ===========================================================================
# Logit lens
# ===========================================================================
@torch.no_grad()
def logit_lens(model, loader, target_class, device, max_batches=None, cls_position=0, save_dir=GRAPH_DIR):
    """
    Observational, layer-level readout: reads each layer's own CLS (and,
    for DeiT-Tiny, DIST) state and projects it through the model's own
    final classifier head, exactly replicating the real prediction rule,
    just applied prematurely at every layer instead of only after the last
    block. Unlike activation patching, no downstream layers actually
    process this early readout, so this describes the model's own
    evolving belief across depth, not the causal effect of an
    intervention. Hardcoded to model.norm, model.cls_head, model.dist_head,
    and model.use_distillation_token.
    """
    was_training = model.training
    model.eval()
    has_dist = getattr(model, "use_distillation_token", False)
    n_layers = len(model.transformer)
    clean_probs = {l: [] for l in range(n_layers)}
    trig_probs = {l: [] for l in range(n_layers)}

    for b_idx, (img_c, _, img_b, _) in enumerate(tqdm(loader, desc="Logit lens")):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        for img, store in [(img_c, clean_probs), (img_b, trig_probs)]:
            layer_states = {}

            def _cap(mod, inp, out, idx):
                layer_states[idx] = out.detach()

            hooks = [blk.register_forward_hook(lambda m, i, o, idx=i_: _cap(m, i, o, idx)) for i_, blk in enumerate(model.transformer)]
            model(img)
            for h in hooks:
                h.remove()
            for l in range(n_layers):
                cls_logits = model.cls_head(model.norm(layer_states[l][:, cls_position, :]))
                if has_dist:
                    dist_logits = model.dist_head(model.norm(layer_states[l][:, 1, :]))
                    logits_l = (cls_logits + dist_logits) / 2
                else:
                    logits_l = cls_logits
                probs_l = torch.softmax(logits_l, dim=-1)[:, target_class]
                store[l].extend(probs_l.detach().cpu().tolist())

    if was_training:
        model.train()
    clean_attribution = {l: float(np.mean(v)) for l, v in clean_probs.items()}
    trig_attribution = {l: float(np.mean(v)) for l, v in trig_probs.items()}
    _plot_logit_lens(clean_attribution, trig_attribution, save_dir)
    return {"clean": clean_attribution, "triggered": trig_attribution}


def _plot_logit_lens(clean_attribution, trig_attribution, save_dir=GRAPH_DIR):
    layers = sorted(clean_attribution.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, [clean_attribution[l] for l in layers], "o-", color="steelblue", label="Clean input")
    ax.plot(layers, [trig_attribution[l] for l in layers], "o-", color="crimson", label="Triggered input")
    ax.set_xlabel("Layer (readout point)"); ax.set_ylabel("P(target class), early readout")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Logit lens: target-class probability from each layer's own state")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_dir / "logit_lens.png", dpi=150); plt.close()


# ===========================================================================
# Clean-model control
# ===========================================================================
def load_model_checkpoint(path, model, device, state_dict_key_candidates=("model", "state_dict", "model_state_dict")):
    """Loads a checkpoint that may be a raw state_dict or wrapped under one of state_dict_key_candidates."""
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and not all(torch.is_tensor(v) for v in raw.values()):
        state = next((raw[k] for k in state_dict_key_candidates if k in raw), None)
        if state is None:
            raise ValueError(f"No state_dict found in checkpoint at {path}, top-level keys: {list(raw.keys())}")
    else:
        state = raw
    model.load_state_dict(state)
    model.to(device); model.eval()
    return model


def diagnose_sample_label_distribution(loader, max_batches, target_class):
    """Checks whether a small max_batches sample is skewed toward one label, since the paired loader is unshuffled."""
    from collections import Counter
    label_counts = Counter()
    n_seen = 0
    for b_idx, (img_c, lbl_c, img_b, lbl_b) in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        label_counts.update(lbl_c.tolist())
        n_seen += len(lbl_c)
    frac_already_target = label_counts.get(target_class, 0) / n_seen if n_seen else 0.0
    if frac_already_target > 0:
        print(f"Warning: {frac_already_target:.1%} of this sample is already labeled target_class, check exclude_same_label.")
    return {"n_seen": n_seen, "label_counts": dict(label_counts), "frac_already_target": frac_already_target}


def run_clean_model_control(clean_model, loader, top_components, r_hat, target_class, device, cls_position=0,
                             max_batches=None, run_activation_patching_control=True, patch_positions=None,
                             backdoored_abs_max=None, save_dir=GRAPH_DIR):
    """
    Rules out generic importance as an alternative explanation: re-runs
    cosine similarity and optionally activation patching on a clean,
    non-backdoored model, for the same components and the same r_hat, to
    check whether their alignment with r_hat is backdoor-specific.
    """
    diagnose_sample_label_distribution(loader, max_batches, target_class)
    clean_head_dirs, clean_mlp_dirs = extract_component_activations(clean_model, loader, device, max_batches=max_batches)
    r = r_hat / (r_hat.norm() + 1e-8)
    comparison = []
    for d in top_components:
        l, comp = d["layer"], d["component"]
        clean_vec = clean_head_dirs[l][int(comp.split("_")[1]), cls_position, :] if comp.startswith("head_") else clean_mlp_dirs[l][cls_position, :]
        clean_sim = F.cosine_similarity(clean_vec.unsqueeze(0), r.unsqueeze(0)).item()
        comparison.append({"label": f"L{l}-{comp}", "backdoored_cos_sim": d["cosine_sim"], "clean_cos_sim": clean_sim})
    _plot_clean_model_control(comparison, save_dir)

    clean_attribution = clean_stderr = None
    if run_activation_patching_control:
        clean_attribution, clean_stderr, _ = activation_patching(
            clean_model, loader, top_components, target_class, device, positions=patch_positions, max_batches=max_batches,
            heatmap_filename="activation_patching_clean_control.png", shared_abs_max=backdoored_abs_max,
            title_suffix=" (clean model control)", save_dir=save_dir)
    return {"cosine_comparison": comparison, "clean_activation_patching": clean_attribution, "clean_activation_patching_stderr": clean_stderr}


def _plot_clean_model_control(comparison, save_dir=GRAPH_DIR):
    labels = [c["label"] for c in comparison]
    backdoored_vals = [c["backdoored_cos_sim"] for c in comparison]
    clean_vals = [c["clean_cos_sim"] for c in comparison]
    diff_vals = [b - c for b, c in zip(backdoored_vals, clean_vals)]

    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(labels) * 1.6), 6.2))
    x = np.arange(len(labels)); width = 0.35
    axes[0].bar(x - width / 2, backdoored_vals, width, label="Backdoored model", color="crimson")
    axes[0].bar(x + width / 2, clean_vals, width, label="Clean model", color="steelblue")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Cosine similarity with r_hat")
    axes[0].set_title("Backdoored vs. clean-model cosine similarity")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, diff_vals, color=["crimson" if v > 0 else "steelblue" for v in diff_vals])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Difference (backdoored - clean)")
    axes[1].set_title("Backdoor-specificity gap per component")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(save_dir / "clean_model_control.png", dpi=150); plt.close()


# ===========================================================================
# Top-level orchestration
# ===========================================================================
def run_phase1_pipeline(model, analysis_loader, directions_cls, r_hat, target_class, device, image_size=32, patch_size=4,
                         cls_offset=1, dynamic_trigger=False, top_k=10, max_batches_extract=None, max_batches_patch=None,
                         patch_positions=None, max_path_pairs=200, max_path_batches=1, max_layer_gap=None,
                         path_patching_position_modes=("cls", "all"), run_full_depth_circuit=True,
                         run_full_depth_activation_patching_flag=True, run_layer_wise_activation_patching_flag=True,
                         run_logit_lens_flag=True, run_necessity_check=True, run_greedy_minimality=True,
                         run_exhaustive_minimality=False, minimality_candidate_components=None,
                         exhaustive_minimality_max_candidates=14, minimality_ra_tolerance=0.02,
                         run_layer_vs_component_defense_proxy=True, run_steering_sweeps=True, n_steering_components=5,
                         steering_alphas=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0), directions_all=None, clean_model=None,
                         clean_model_max_batches=None):
    """
    Runs the full Phase 1 pipeline: sanity check, component extraction and
    ranking, logit lens, activation patching, layer-wise sufficiency,
    functional roles, path patching, necessity patching, minimal circuit
    search, defense proxy, steering sweeps, and an optional clean-model
    control. Every step past the sanity check is individually toggleable.
    """
    if not sanity_check_manual_attention(model, analysis_loader, device):
        raise RuntimeError("Manual attention reconstruction does not match the real forward pass, fix before proceeding.")

    fixed_trigger_position = compute_trigger_token_position(image_size, patch_size, cls_offset=cls_offset)
    head_dirs, mlp_dirs = extract_component_activations(model, analysis_loader, device, max_batches=max_batches_extract)
    verify_extraction_consistency(head_dirs, mlp_dirs, directions_cls)
    top_components, all_scored = rank_components_by_cosine_similarity(head_dirs, mlp_dirs, r_hat, top_k=top_k, plot=(clean_model is None))
    plot_cosine_similarity_distribution(all_scored, top_k)

    attribution, attribution_stderr, backdoored_abs_max = activation_patching(
        model, analysis_loader, top_components, target_class, device, positions=patch_positions, max_batches=max_batches_patch)
    n_tokens = model.pos_embedding.shape[1]
    positions_used = patch_positions if patch_positions is not None else list(range(n_tokens))
    component_list = [(d["layer"], d["component"]) for d in top_components]
    strongest = identify_strongest_position_per_component(attribution, component_list, positions_used)

    logit_lens_results = None
    if run_logit_lens_flag:
        logit_lens_results = logit_lens(model, analysis_loader, target_class, device, max_batches=max_batches_extract)

    full_depth_activation_results = None
    if run_full_depth_activation_patching_flag:
        full_depth_attribution, full_depth_ap_components = run_full_depth_activation_patching(
            model, analysis_loader, all_scored, target_class, device, max_batches=max_batches_patch)
        full_depth_activation_results = {"attribution": full_depth_attribution, "components": full_depth_ap_components}

    layer_wise_results = None
    if run_layer_wise_activation_patching_flag:
        layer_wise_attribution = layer_wise_activation_patching(model, analysis_loader, target_class, device, max_batches=max_batches_patch)
        comparison = compare_layer_sufficiency_vs_component_sum(layer_wise_attribution, attribution, top_components)
        layer_wise_results = {"layer_attribution": layer_wise_attribution, "comparison": comparison}

    trig_arg = None if dynamic_trigger else fixed_trigger_position
    attn_to_trigger = compute_attention_to_trigger(model, analysis_loader, device, trigger_position=trig_arg,
                                                    patch_size=patch_size, cls_offset=cls_offset, max_batches=max_batches_extract)
    role_results = classify_functional_roles(attn_to_trigger, attribution, top_components,
                                              full_depth_attribution=(full_depth_activation_results["attribution"] if full_depth_activation_results else None))
    role_map = {r["label"]: r["role"] for r in role_results}

    edge_scores_by_mode = run_path_patching(model, analysis_loader, top_components, target_class, device, max_pairs=max_path_pairs,
                                             max_batches=max_path_batches, max_layer_gap=max_layer_gap,
                                             position_modes=path_patching_position_modes, role_map=role_map)

    full_depth_path_results = None
    if run_full_depth_circuit:
        full_depth_edges, full_depth_pp_components = run_full_depth_path_patching(
            model, analysis_loader, all_scored, target_class, device, max_batches=max_path_batches, role_map=role_map)
        full_depth_path_results = {"edges": full_depth_edges, "components": full_depth_pp_components}

    necessity_results = None
    if run_necessity_check:
        necessity_results = necessity_patching(model, analysis_loader, top_components, target_class, device, max_batches=max_batches_patch)

    minimality_candidates = minimality_candidate_components if minimality_candidate_components is not None else top_components
    greedy_results = {}
    if run_greedy_minimality:
        for mode in ("cls", "all"):
            greedy_results[mode] = greedy_minimal_circuit_search(model, analysis_loader, minimality_candidates, target_class, device,
                                                                   position_mode=mode, max_batches=max_batches_patch, ra_tolerance=minimality_ra_tolerance)

    exhaustive_results = {}
    if run_exhaustive_minimality:
        for mode in ("cls", "all"):
            exhaustive_results[mode] = exhaustive_minimal_circuit_search(model, analysis_loader, minimality_candidates, target_class, device,
                                                                           position_mode=mode, max_batches=max_batches_patch,
                                                                           ra_tolerance=minimality_ra_tolerance, max_candidates=exhaustive_minimality_max_candidates)

    if run_greedy_minimality or run_exhaustive_minimality:
        for mode in ("cls", "all"):
            plot_all_minimal_circuits(minimality_candidates, greedy_results.get(mode), exhaustive_results.get(mode), role_map=role_map)

    defense_proxy_results = None
    if run_layer_vs_component_defense_proxy:
        minimal_labels = None
        for mode in ("all", "cls"):
            if exhaustive_results.get(mode) and exhaustive_results[mode]["minimal_size"] is not None:
                sz = exhaustive_results[mode]["minimal_size"]
                minimal_labels = exhaustive_results[mode]["best_by_size"][sz]["combo"]
                break
            if greedy_results.get(mode) and greedy_results[mode]["minimal_set_size"] is not None:
                sz = greedy_results[mode]["minimal_set_size"]
                minimal_labels = greedy_results[mode]["history"][sz - 1]["cumulative_set"]
                break
        if minimal_labels:
            component_set = [(int(lbl[1:].split("-", 1)[0]), lbl[1:].split("-", 1)[1]) for lbl in minimal_labels]
            defense_proxy_all = compare_layer_vs_component_defense_proxy(model, analysis_loader, target_class, device, r_hat,
                                                                          component_set, token_scope="all", max_batches=max_batches_patch)
            defense_proxy_cls = compare_layer_vs_component_defense_proxy(model, analysis_loader, target_class, device, r_hat,
                                                                          component_set, token_scope="cls", max_batches=max_batches_patch)
            defense_proxy_results = {"all": defense_proxy_all, "cls_diagnostic": defense_proxy_cls}

    steering_sweep_results = None
    if run_steering_sweeps:
        best = top_components[0]
        layer_direction_cls = directions_cls[best["layer"]]
        layer_direction_all = directions_all[best["layer"]].view(n_tokens, -1) if directions_all is not None else None
        steering_sweep_results = sweep_multiple_components_steering_magnitude(
            model, analysis_loader, top_components, head_dirs, mlp_dirs, layer_direction_cls, best["layer"], target_class, device,
            n_components=n_steering_components, alphas=steering_alphas, layer_reference_direction_all=layer_direction_all, max_batches=max_batches_patch)

    clean_control_results = None
    if clean_model is not None:
        clean_control_results = run_clean_model_control(clean_model, analysis_loader, top_components, r_hat, target_class, device,
                                                         max_batches=clean_model_max_batches or max_batches_extract,
                                                         patch_positions=patch_positions, backdoored_abs_max=backdoored_abs_max)

    return {
        "top_components": top_components, "all_scored_components": all_scored,
        "activation_patching_attribution": attribution, "activation_patching_stderr": attribution_stderr,
        "logit_lens": logit_lens_results, "full_depth_activation_patching": full_depth_activation_results,
        "layer_wise_activation_patching": layer_wise_results, "strongest_position_per_component": strongest,
        "path_patching_edges_by_mode": edge_scores_by_mode, "full_depth_path_patching": full_depth_path_results,
        "necessity_patching": necessity_results, "greedy_minimality_search": greedy_results,
        "exhaustive_minimality_search": exhaustive_results, "layer_vs_component_defense_proxy": defense_proxy_results,
        "functional_roles": role_results, "steering_sweep_results": steering_sweep_results, "clean_model_control": clean_control_results,
    }