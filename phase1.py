"""
Phase 1: component-level circuit analysis for backdoor detection in ViTs.

Assumes phase0.py has already run and produced a backdoor-trained model,
directions_cls (r^l for every layer), and r_hat (the causally validated
most representative direction).

All functions take the same 4-output paired loader as phase0:
(clean_img, clean_label, backdoor_img, backdoor_label).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

GRAPH_DIR = Path("./results/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def get_graph_dir(model_name, attack_type, dataset, base="results"):
    """
    Builds and returns results/{model_name}/{attack_type}/{dataset}.
    Pass the result as run_phase1_pipeline's save_dir so each
    model/attack/dataset combination writes to its own folder.
    """
    path = Path(base) / model_name / attack_type / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


# Plot order follows the narrative: direction correlation and specificity
# first, then observational corroboration, then causal validation.
PLOT_ORDER = {
    "cosine_similarity_full_grid": 1,
    "direction_consistency_matrix": 2,
    "direction_alignment_with_rhat_backdoored": 3,
    "clean_model_control": 4,
    "logit_lens": 5,
    "steering_layer": 6,
    "cumulative_component_steering": 7,
}


def _numbered_path(save_dir, stem):
    """Builds save_dir/NN_stem.png using the longest matching PLOT_ORDER key, so mode-suffixed filenames share a number and sort together."""
    matches = [k for k in PLOT_ORDER if stem == k or stem.startswith(k + "_")]
    key = max(matches, key=len) if matches else None
    n = PLOT_ORDER.get(key, 99)
    return Path(save_dir) / f"{n:02d}_{stem}.png"


def _manual_per_head_attention(attn_module, x):
    """
    Reimplements nn.MultiheadAttention's math by hand so each head's own
    contribution can be isolated. The built-in module only exposes the
    summed output across heads, but steering needs to modify one head at
    a time.

    Args:
        attn_module: an nn.MultiheadAttention module.
        x: input tensor (B, N, D).

    Returns:
        head_contribs: (B, H, N, D) tensor, each head's own write to the
            residual stream, before summing across heads and before
            out_proj.bias.
        attn_weights: (B, H, N, N) softmax attention pattern.
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
    Confirms that _manual_per_head_attention's per-head reconstruction
    sums back to the model's real attention output.

    Args:
        model: the ViT model.
        loader: paired loader, only the clean image from the first batch is used.
        device: torch device.
        atol: maximum allowed reconstruction error.

    Returns:
        True if the check passed, False otherwise.
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
          f"({'PASSED' if ok else 'FAILED, fix before trusting any steering result'})")
    return ok


@torch.no_grad()
def extract_component_activations(model, loader, device, max_batches=None):
    """
    Computes the mean triggered-minus-clean difference (r^c) for every
    attention head and MLP block, at every token position. These same
    direction vectors are used for steering here and for Phase 2's
    weight orthogonalisation.

    Args:
        model: the ViT model.
        loader: paired loader.
        device: torch device.
        max_batches: optional cap on how many batches to process.

    Returns:
        head_dirs: dict layer -> tensor (H, N, D).
        mlp_dirs: dict layer -> tensor (N, D).
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
    Bug check: since the residual stream is a pure additive
    accumulator, r^l minus r^(l-1) should equal the sum of every head's
    and the MLP's own write at that layer. A mismatch here means phase0
    and phase1 were extracted from different samples or a hook is wrong.

    Args:
        head_dirs, mlp_dirs: from extract_component_activations.
        directions_cls: dict layer -> r^l from phase0.
        cls_position: which token position to check.
        tol: maximum allowed relative residual.

    Returns:
        True if consistent, False otherwise.
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


def rank_components_by_cosine_similarity(head_dirs, mlp_dirs, r_hat, cls_position=0, top_k=10):
    """
    Ranks every component by cosine similarity of its own r^c (at the CLS
    position) against r_hat.

    Args:
        head_dirs, mlp_dirs: from extract_component_activations.
        r_hat: the globally chosen direction from phase0.
        cls_position: which token position to compare.
        top_k: how many top components to return.

    Returns:
        top: the top_k scored components, sorted by |cosine_sim| descending.
        scored: every component, same sort order.
    """
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
    return scored[:top_k], scored


def plot_cosine_similarity_full_grid(all_scored, top_k=None, save_dir=GRAPH_DIR):
    """
    Plots cosine similarity with r_hat for every component in the model,
    as a layer by component-slot grid. Reuses rank_components_by_cosine_similarity's
    own output, no extra forward passes needed.

    Args:
        all_scored: the full scored list from rank_components_by_cosine_similarity.
        top_k: if given, outlines the top_k cells in black.
        save_dir: where to save the plot.
    """
    layers = sorted(set(d["layer"] for d in all_scored))
    head_slots = sorted({d["component"] for d in all_scored if d["component"].startswith("head_")}, key=lambda s: int(s.split("_")[1]))
    slots = head_slots + (["mlp"] if any(d["component"] == "mlp" for d in all_scored) else [])
    mat = np.full((len(slots), len(layers)), np.nan)
    for d in all_scored:
        mat[slots.index(d["component"]), layers.index(d["layer"])] = d["cosine_sim"]

    top_keys = set()
    if top_k:
        ranked = sorted(all_scored, key=lambda d: abs(d["cosine_sim"]), reverse=True)[:top_k]
        top_keys = {(d["layer"], d["component"]) for d in ranked}

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(layers) + 3), max(4, 0.4 * len(slots) + 2)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(layers))); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_yticks(range(len(slots))); ax.set_yticklabels(slots, fontsize=8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Component slot")
    for i, slot in enumerate(slots):
        for j, l in enumerate(layers):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
            if (l, slot) in top_keys:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2))
    plt.colorbar(im, ax=ax, label="Cosine similarity of r^c with r_hat")
    title = "Cosine similarity of each component's own r^c with r_hat"
    if top_k:
        title += f", top {top_k} outlined"
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(_numbered_path(save_dir, "cosine_similarity_full_grid"), dpi=150); plt.close()


def plot_direction_consistency_matrix(directions_cls, save_dir=GRAPH_DIR):
    """
    Plots cosine similarity between every pair of layers' own r^l. Checks
    whether the backdoor is written along one persistent direction as
    depth increases (values near 1 everywhere) or re-encoded at different
    depths (values well below 1 off the diagonal).

    Args:
        directions_cls: dict layer -> r^l.
        save_dir: where to save the plot.

    Returns:
        the NxN matrix as a numpy array.
    """
    layers = sorted(directions_cls.keys())
    n = len(layers)
    mat = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            a, b = directions_cls[li], directions_cls[lj]
            mat[i, j] = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n), max(5, 0.6 * n)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("r^l cosine similarity across layers")
    plt.tight_layout(); plt.savefig(_numbered_path(save_dir, "direction_consistency_matrix"), dpi=150); plt.close()
    return mat


def compute_direction_alignment_with_rhat(directions_cls, r_hat, save_dir=GRAPH_DIR):
    """
    Splits each layer's own r^l into a part parallel to r_hat and a
    residual orthogonal to it. Cosine similarity only measures angle and
    ignores scale, so this reports how much of r^l's actual magnitude is
    captured by r_hat.

    Args:
        directions_cls: dict layer -> r^l.
        r_hat: the globally chosen direction from phase0.
        save_dir: where to save the plot.

    Returns:
        dict with layers, parallel_magnitude, and residual_fraction per layer.
    """
    r_unit = r_hat / (r_hat.norm() + 1e-8)
    layers = sorted(directions_cls.keys())
    residual_fracs, parallel_mags = [], []
    for l in layers:
        r_l = directions_cls[l]
        parallel_mag = (r_l * r_unit).sum().item()
        residual = r_l - parallel_mag * r_unit
        total_norm = r_l.norm().item() + 1e-8
        residual_fracs.append(residual.norm().item() / total_norm)
        parallel_mags.append(parallel_mag)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar([str(l) for l in layers], parallel_mags, color="crimson")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Projection of r^l onto r_hat")
    axes[0].set_title("r^l projected onto r_hat, by layer")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar([str(l) for l in layers], residual_fracs, color="steelblue")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Fraction of r^l orthogonal to r_hat")
    axes[1].set_title("Fraction of r^l not explained by r_hat")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(_numbered_path(save_dir, "direction_alignment_with_rhat_backdoored"), dpi=150); plt.close()
    return {"layers": layers, "parallel_magnitude": parallel_mags, "residual_fraction": residual_fracs}


def component_activation_steering(model, loader, layer_idx, component, direction, mode, target_class, device,
                                   alpha=1.0, max_batches=None, steer_all_tokens=False, token_position=0):
    """
    Adds or subtracts alpha times direction to one component's own
    output, and reports both ASR and the recovery metric from the same
    forward pass.

    Args:
        model: the ViT model.
        loader: paired loader.
        layer_idx: which layer.
        component: "head_h" or "mlp".
        direction: the vector to steer by.
        mode: "positive" adds the direction to clean inputs.
            "negative" subtracts it from triggered inputs.
        target_class: the attack's target class.
        device: torch device.
        alpha: steering magnitude.
        max_batches: optional cap.
        steer_all_tokens: if True, steers every token position, ignoring token_position.
        token_position: which single token to steer when steer_all_tokens
            is False. 0 is CLS, 1 is DIST for DeiT-Tiny.

    Returns:
        asr: fraction of predictions equal to target_class.
        second: fraction of predictions equal to the true label, ACC on
            clean images (mode="positive") or RA on triggered images
            (mode="negative").
    """
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
                head_contribs[:, h, token_position, :] += sign * alpha * d
            patched = head_contribs.sum(dim=1)
            if mod.out_proj.bias is not None:
                patched = patched + mod.out_proj.bias
            return (patched, out[1] if isinstance(out, tuple) else None)
        patched = out.clone()
        if steer_all_tokens:
            patched += sign * alpha * d.unsqueeze(0)
        else:
            patched[:, token_position, :] += sign * alpha * d
        return patched

    blk = model.transformer[layer_idx]
    target_module = blk.attn if component.startswith("head_") else blk.mlp
    hook_handle = target_module.register_forward_hook(hook)
    target_hits = true_hits = total = 0
    with torch.no_grad():
        for b_idx, (img_c, lbl_c, img_b, _) in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
            inputs = img_c.to(device) if mode == "positive" else img_b.to(device)
            preds = model(inputs).argmax(1).cpu()
            target_hits += (preds == target_class).sum().item()
            true_hits += (preds == lbl_c).sum().item()
            total += len(img_c)
    hook_handle.remove()
    asr = target_hits / total if total else 0.0
    second = true_hits / total if total else 0.0
    return asr, second


def sweep_all_heads_steering(model, loader, head_dirs, mlp_dirs, target_class, device,
                              alphas=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0), max_batches=None,
                              dist_position=None, save_dir=GRAPH_DIR):
    """
    Breadth sweep: for every layer, plots every component's magnitude
    sweep steering curve alongside that layer's own whole-layer
    reference curve. Four rows per position mode: ASR and ACC for
    positive steering (added to clean images), ASR and RA for negative
    steering (subtracted from triggered images), covering CLS-only and
    all-tokens, plus DIST-only when dist_position is given (DeiT-Tiny).
    One file per layer, 4 rows times 2 or 3 columns. This is the
    exhaustive, unranked scan, complementing cumulative_component_steering
    below which is targeted and cosine-ranked instead.

    Args:
        model: the ViT model.
        loader: paired loader.
        head_dirs, mlp_dirs: from extract_component_activations.
        target_class: the attack's target class.
        device: torch device.
        alphas: steering magnitudes to sweep.
        max_batches: optional cap per evaluation.
        dist_position: token position of the DIST token. Leave None to
            skip DIST mode entirely, matching the previous behaviour
            exactly.
        save_dir: where to save plots.

    Returns:
        dict layer -> results, each with cls, all, and if applicable
        dist mode curves per component.
    """
    n_layers = len(model.transformer)
    H = head_dirs[0].shape[0]
    all_results = {}

    def _steer_whole_layer(layer_idx, direction, mode, alpha, steer_all_tokens, token_position=0):
        d = direction.to(device)
        sign = 1 if mode == "positive" else -1

        def hook(mod, inp, out):
            steered = out.clone()
            if steer_all_tokens:
                steered = steered + sign * alpha * d.unsqueeze(0)
            else:
                steered[:, token_position, :] += sign * alpha * d
            return steered

        h = model.transformer[layer_idx].register_forward_hook(hook)
        target_hits = true_hits = total = 0
        with torch.no_grad():
            for b_idx, (img_c, lbl_c, img_b, _) in enumerate(loader):
                if max_batches and b_idx >= max_batches:
                    break
                inputs = img_c.to(device) if mode == "positive" else img_b.to(device)
                preds = model(inputs).argmax(1).cpu()
                target_hits += (preds == target_class).sum().item()
                true_hits += (preds == lbl_c).sum().item()
                total += len(img_c)
        h.remove()
        asr = target_hits / total if total else 0.0
        second = true_hits / total if total else 0.0
        return asr, second

    n_cols = 3 if dist_position is not None else 2

    for l in tqdm(range(n_layers), desc="Full-model steering sweep"):
        components = [f"head_{h}" for h in range(H)] + ["mlp"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
        layer_dir_cls = head_dirs[l][:, 0, :].sum(0) + mlp_dirs[l][0, :]
        layer_dir_all = head_dirs[l].sum(0) + mlp_dirs[l]

        fig, axes = plt.subplots(4, n_cols, figsize=(8 * n_cols, 20))
        layer_results = {"cls": {}, "all": {}}
        if dist_position is not None:
            layer_results["dist"] = {}

        modes = [("cls", False, layer_dir_cls, 0), ("all", True, layer_dir_all, 0)]
        if dist_position is not None:
            layer_dir_dist = head_dirs[l][:, dist_position, :].sum(0) + mlp_dirs[l][dist_position, :]
            modes.append(("dist", False, layer_dir_dist, dist_position))

        for col, (mode_key, steer_all, layer_dir, token_position) in enumerate(modes):
            ax_asr_pos, ax_acc_pos = axes[0, col], axes[1, col]
            ax_asr_neg, ax_ra_neg = axes[2, col], axes[3, col]

            layer_asr_pos, layer_acc_pos = zip(*[_steer_whole_layer(l, layer_dir, "positive", a, steer_all, token_position) for a in alphas])
            layer_asr_neg, layer_ra_neg = zip(*[_steer_whole_layer(l, layer_dir, "negative", a, steer_all, token_position) for a in alphas])
            for ax, curve in [(ax_asr_pos, layer_asr_pos), (ax_acc_pos, layer_acc_pos),
                              (ax_asr_neg, layer_asr_neg), (ax_ra_neg, layer_ra_neg)]:
                ax.plot(alphas, curve, "k--", lw=2.5, marker="D", label=f"Layer {l} reference")
            layer_results[mode_key]["layer_reference"] = {
                "asr_pos": list(layer_asr_pos), "acc_pos": list(layer_acc_pos),
                "asr_neg": list(layer_asr_neg), "ra_neg": list(layer_ra_neg),
            }

            for idx, comp in enumerate(components):
                if comp.startswith("head_"):
                    h_idx = int(comp.split("_")[1])
                    direction = head_dirs[l][h_idx] if steer_all else head_dirs[l][h_idx, token_position, :]
                else:
                    direction = mlp_dirs[l] if steer_all else mlp_dirs[l][token_position, :]

                asr_pos, acc_pos = zip(*[component_activation_steering(model, loader, l, comp, direction, "positive", target_class, device, a, max_batches, steer_all, token_position) for a in alphas])
                asr_neg, ra_neg = zip(*[component_activation_steering(model, loader, l, comp, direction, "negative", target_class, device, a, max_batches, steer_all, token_position) for a in alphas])

                ax_asr_pos.plot(alphas, asr_pos, "o-", color=colors[idx], label=comp, markersize=4)
                ax_acc_pos.plot(alphas, acc_pos, "o-", color=colors[idx], label=comp, markersize=4)
                ax_asr_neg.plot(alphas, asr_neg, "o-", color=colors[idx], label=comp, markersize=4)
                ax_ra_neg.plot(alphas, ra_neg, "o-", color=colors[idx], label=comp, markersize=4)
                layer_results[mode_key][comp] = {
                    "asr_pos": list(asr_pos), "acc_pos": list(acc_pos),
                    "asr_neg": list(asr_neg), "ra_neg": list(ra_neg),
                }

            scope = {"cls": "CLS only", "all": "all tokens", "dist": "DIST only"}[mode_key]
            ax_asr_pos.set_title(f"ASR (Positive Steering), {scope}")
            ax_acc_pos.set_title(f"ACC (Positive Steering), {scope}")
            ax_asr_neg.set_title(f"ASR (Negative Steering), {scope}")
            ax_ra_neg.set_title(f"RA (Negative Steering), {scope}")

        for ax in axes.flat:
            ax.axvline(1.0, color="grey", ls=":", alpha=0.5)
            ax.set_xlabel("Steering magnitude"); ax.set_ylabel("Rate")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Layer {l}: steering with each component's own r^c, vs whole-layer reference", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(_numbered_path(save_dir, f"steering_layer_{l:02d}"), dpi=150, bbox_inches="tight")
        plt.close()
        all_results[l] = layer_results

    return all_results


def _steer_component_set(model, loader, component_set, component_directions, apply_to, target_class, device,
                          sign=1, alpha=1.0, max_batches=None, steer_all_tokens=False):
    """
    Steers every component in component_set at once, each by its own
    direction, in one forward pass per batch. Components at the same
    layer share a single hook.

    Args:
        model: the ViT model.
        loader: paired loader.
        component_set: list of (layer, component) tuples to steer together.
        component_directions: dict (layer, component) -> direction vector.
        apply_to: "triggered" or "clean", which images to run.
        target_class: the attack's target class.
        device: torch device.
        sign: +1 to add the direction, -1 to subtract it.
        alpha: steering magnitude.
        max_batches: optional cap.
        steer_all_tokens: if True, steers every token position, otherwise
            just the CLS position.

    Returns:
        asr: fraction of predictions equal to target_class.
        acc: fraction of predictions equal to the true label.
    """
    by_layer_attn, by_layer_mlp = {}, {}
    for (l, c) in component_set:
        d = component_directions[(l, c)].to(device)
        if c.startswith("head_"):
            by_layer_attn.setdefault(l, []).append((int(c.split("_")[1]), d))
        else:
            by_layer_mlp[l] = d

    hooks = []
    for l, head_list in by_layer_attn.items():
        def attn_hook(mod, inp, out, head_list=head_list):
            head_contribs, _ = _manual_per_head_attention(mod, inp[0])
            head_contribs = head_contribs.clone()
            for h, d in head_list:
                if steer_all_tokens:
                    head_contribs[:, h, :, :] += sign * alpha * d.unsqueeze(0)
                else:
                    head_contribs[:, h, 0, :] += sign * alpha * d
            patched = head_contribs.sum(dim=1)
            if mod.out_proj.bias is not None:
                patched = patched + mod.out_proj.bias
            return (patched, out[1] if isinstance(out, tuple) else None)
        hooks.append(model.transformer[l].attn.register_forward_hook(attn_hook))

    for l, d in by_layer_mlp.items():
        def mlp_hook(mod, inp, out, d=d):
            patched = out.clone()
            if steer_all_tokens:
                patched += sign * alpha * d.unsqueeze(0)
            else:
                patched[:, 0, :] += sign * alpha * d
            return patched
        hooks.append(model.transformer[l].mlp.register_forward_hook(mlp_hook))

    target_hits = true_hits = total = 0
    with torch.no_grad():
        for b_idx, (img_c, lbl_c, img_b, _) in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
            inputs = img_b.to(device) if apply_to == "triggered" else img_c.to(device)
            preds = model(inputs).argmax(1).cpu()
            target_hits += (preds == target_class).sum().item()
            true_hits += (preds == lbl_c).sum().item()
            total += len(img_c)
    for h in hooks:
        h.remove()

    asr = target_hits / total if total else 0.0
    acc = true_hits / total if total else 0.0
    return asr, acc


@torch.no_grad()
def _eval_baselines(model, loader, target_class, device, max_batches=None):
    """
    Computes the unsteered reference values used on the cumulative
    steering plot: ASR and RA on triggered images, and both accuracy and
    ASR on clean images.

    Args:
        model: the ViT model.
        loader: paired loader.
        target_class: the attack's target class.
        device: torch device.
        max_batches: optional cap.

    Returns:
        baseline_asr, baseline_ra, baseline_clean_acc, baseline_clean_asr as fractions.
    """
    model.eval()
    target_hits = true_hits = clean_hits = clean_target_hits = total = 0
    for b_idx, (img_c, lbl_c, img_b, _) in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        img_c, img_b = img_c.to(device), img_b.to(device)
        preds_b = model(img_b).argmax(1).cpu()
        preds_c = model(img_c).argmax(1).cpu()
        target_hits += (preds_b == target_class).sum().item()
        true_hits += (preds_b == lbl_c).sum().item()
        clean_hits += (preds_c == lbl_c).sum().item()
        clean_target_hits += (preds_c == target_class).sum().item()
        total += len(img_c)
    if not total:
        return 0.0, 0.0, 0.0, 0.0
    return target_hits / total, true_hits / total, clean_hits / total, clean_target_hits / total


def cumulative_component_steering(model, loader, all_scored_components, head_dirs, mlp_dirs, target_class, device,
                                   alpha=1.0, cls_position=0, max_batches=None,
                                   thresholds=None, save_dir=GRAPH_DIR):
    """
    The steering analog of a minimal-circuit search, grouped by cosine
    similarity threshold rather than added one component at a time.
    Starting from threshold 1.0 and stepping down in increments (0.05
    by default) to 0.0, each step's component set is every component
    whose own |cosine_sim| with r_hat is at least that threshold. Since
    |cosine_sim| is always in [0, 1], a threshold of 0.0 includes every
    component in the model, so this always ends at full coverage. Signed
    cosine similarity is not used for the threshold: a strongly
    anti-aligned component (cosine similarity close to -1) is just as
    causally relevant as a strongly aligned one.

    At each threshold, runs the same two interventions in both modes:

    Necessity: every component in the current set has its own direction
    subtracted, jointly, from triggered images. Tests whether removing
    the cumulative backdoor push from just this threshold's components
    restores clean behaviour.

    Sufficiency: every component in the current set has its own
    direction added, jointly, to clean images. Tests whether injecting
    just this threshold's components' combined push alone flips clean
    predictions, and how much collateral damage that injection does to
    otherwise correct clean predictions.

    component_directions_cls is always the flat, CLS-position version,
    exactly what component_level_orthogonalize expects, so this
    function's output is the candidate set and direction source for
    Phase 2's weight-space defence. component_directions is the all-tokens 
    version, used only for the all-tokens steering test here, and cannot 
    be used for orthogonalisation.

    Args:
        model: the ViT model.
        loader: paired loader.
        all_scored_components: the full scored list from
            rank_components_by_cosine_similarity's second return value.
        head_dirs, mlp_dirs: from extract_component_activations.
        target_class: the attack's target class.
        device: torch device.
        alpha: steering magnitude.
        cls_position: which token position counts as CLS.
        max_batches: optional cap per evaluation.
        thresholds: cosine similarity thresholds to step through,
            descending. Defaults to 1.0, 0.95, ..., 0.0 in 0.05
            increments, 21 steps.
        save_dir: where to save the plot.

    Returns:
        dict with history_cls, history_all (per-threshold results for
        each mode), component_directions, component_directions_cls, the
        four baseline values, and total_components.
    """
    if thresholds is None:
        thresholds = [round(1.0 - 0.05 * i, 2) for i in range(21)]

    component_directions = {}
    component_directions_cls = {}
    for d in all_scored_components:
        l, c = d["layer"], d["component"]
        if c.startswith("head_"):
            h = int(c.split("_")[1])
            component_directions[(l, c)] = head_dirs[l][h]
            component_directions_cls[(l, c)] = head_dirs[l][h, cls_position, :]
        else:
            component_directions[(l, c)] = mlp_dirs[l]
            component_directions_cls[(l, c)] = mlp_dirs[l][cls_position, :]

    total_components = len(all_scored_components)
    baseline_asr, baseline_ra, baseline_clean_acc, baseline_clean_asr = _eval_baselines(model, loader, target_class, device, max_batches)

    def _run_sweep(directions_dict, steer_all_tokens, desc):
        history = []
        prev_set_key, prev_result = None, None
        for t in tqdm(thresholds, desc=desc):
            cumulative_set = [(d["layer"], d["component"]) for d in all_scored_components if abs(d["cosine_sim"]) >= t]
            set_key = frozenset(cumulative_set)

            if set_key == prev_set_key:
                nec_asr, nec_ra, suf_asr, suf_acc = prev_result
            else:
                nec_asr, nec_ra = _steer_component_set(
                    model, loader, cumulative_set, directions_dict, "triggered", target_class, device,
                    sign=-1, alpha=alpha, max_batches=max_batches, steer_all_tokens=steer_all_tokens)
                suf_asr, suf_acc = _steer_component_set(
                    model, loader, cumulative_set, directions_dict, "clean", target_class, device,
                    sign=+1, alpha=alpha, max_batches=max_batches, steer_all_tokens=steer_all_tokens)
                prev_result = (nec_asr, nec_ra, suf_asr, suf_acc)
                prev_set_key = set_key

            history.append({
                "threshold": t, "set_size": len(cumulative_set),
                "pct_components": 100.0 * len(cumulative_set) / total_components if total_components else 0.0,
                "cumulative_set": cumulative_set,
                "necessity_asr": nec_asr, "necessity_ra": nec_ra,
                "sufficiency_asr": suf_asr, "sufficiency_acc": suf_acc,
            })
        return history

    history_cls = _run_sweep(component_directions_cls, steer_all_tokens=False, desc="Cumulative steering, CLS only")
    history_all = _run_sweep(component_directions, steer_all_tokens=True, desc="Cumulative steering, all tokens")

    _plot_cumulative_component_steering(history_cls, history_all, baseline_asr, baseline_ra, baseline_clean_acc,
                                         baseline_clean_asr, total_components, save_dir)
    return {"history_cls": history_cls, "history_all": history_all,
            "component_directions": component_directions,
            "component_directions_cls": component_directions_cls,
            "baseline_asr": baseline_asr, "baseline_ra": baseline_ra,
            "baseline_clean_acc": baseline_clean_acc, "baseline_clean_asr": baseline_clean_asr,
            "total_components": total_components}


def _plot_cumulative_component_steering(history_cls, history_all, baseline_asr, baseline_ra, baseline_clean_acc,
                                         baseline_clean_asr, total_components, save_dir=GRAPH_DIR):
    """
    Two rows, three columns: top row is CLS-only, bottom row is
    all-tokens, matching sweep_all_heads_steering's layout style.
    Columns are necessity, sufficiency, and coverage, same as before.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))

    for row, (history, scope) in enumerate([(history_cls, "CLS only"), (history_all, "all tokens")]):
        thresholds = [h["threshold"] for h in history]

        ax = axes[row, 0]
        ax.plot(thresholds, [h["necessity_ra"] for h in history], "go-", lw=2, markersize=6, label="RA")
        ax.plot(thresholds, [h["necessity_asr"] for h in history], "ro-", lw=2, markersize=6, label="ASR")
        ax.axhline(baseline_ra, color="green", ls="--", alpha=0.5, label=f"Unsteered RA = {baseline_ra:.3f}")
        ax.axhline(baseline_asr, color="red", ls="--", alpha=0.5, label=f"Unsteered ASR = {baseline_asr:.3f}")
        ax.set_ylabel("Rate"); ax.set_ylim(-0.05, 1.08)
        ax.set_title(f"Negative steering (necessity): r^c subtracted from triggered images, {scope}")

        ax = axes[row, 1]
        ax.plot(thresholds, [h["sufficiency_acc"] for h in history], "go-", lw=2, markersize=6, label="ACC")
        ax.plot(thresholds, [h["sufficiency_asr"] for h in history], "ro-", lw=2, markersize=6, label="ASR")
        ax.axhline(baseline_clean_acc, color="green", ls="--", alpha=0.5, label=f"Unsteered ACC = {baseline_clean_acc:.3f}")
        ax.axhline(baseline_clean_asr, color="red", ls="--", alpha=0.5, label=f"Unsteered ASR = {baseline_clean_asr:.3f}")
        ax.set_ylabel("Rate"); ax.set_ylim(-0.05, 1.08)
        ax.set_title(f"Positive steering (sufficiency): r^c added to clean images, {scope}")

        for ax in axes[row, :2]:
            ax.set_xlabel("Cosine similarity threshold (|cosine_sim| >= threshold)")
            ax.invert_xaxis()
            ax.set_xticks([round(t, 1) for t in np.arange(1.0, -0.01, -0.1)])
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(True, alpha=0.3)

        ax = axes[row, 2]
        ax.step(thresholds, [h["set_size"] for h in history], where="post", color="steelblue", lw=2)
        ax.set_xlabel("Cosine similarity threshold (|cosine_sim| >= threshold)")
        ax.set_ylabel("Number of components included")
        ax.set_ylim(-0.5, total_components + 0.5)
        ax.invert_xaxis()
        ax.set_xticks([round(t, 1) for t in np.arange(1.0, -0.01, -0.1)])
        ax.set_title(f"Coverage: number of components included, out of {total_components} total, {scope}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cumulative steering by cosine similarity threshold, using each component's own r^c",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(_numbered_path(save_dir, "cumulative_component_steering"), dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def logit_lens(model, loader, target_class, device, max_batches=None, cls_position=0, save_dir=GRAPH_DIR):
    """
    Reads each layer's own CLS state (and DIST state for DeiT-Tiny) and
    projects it through the model's own final classifier head, at every
    layer instead of only after the last block. This is observational,
    not causal: no downstream layer actually processes this early
    readout. It describes the model's own evolving belief across depth,
    useful as independent corroboration for whichever layer steering
    flags as most representative.

    Args:
        model: the ViT model.
        loader: paired loader.
        target_class: the attack's target class.
        device: torch device.
        max_batches: optional cap.
        cls_position: which token position counts as CLS.
        save_dir: where to save the plot.

    Returns:
        dict with clean and triggered target-class probability per layer.
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
    ax.set_xlabel("Layer, readout point"); ax.set_ylabel("P(target class), early readout")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Logit lens: target-class probability from each layer's own state")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(_numbered_path(save_dir, "logit_lens"), dpi=150); plt.close()


def load_model_checkpoint(path, model, device, state_dict_key_candidates=("model", "state_dict", "model_state_dict")):
    """
    Loads a checkpoint that may be a raw state dict or wrapped under one
    of state_dict_key_candidates.

    Args:
        path: checkpoint file path.
        model: the model instance to load into.
        device: torch device to move the model to.
        state_dict_key_candidates: keys to check if the checkpoint is wrapped.

    Returns:
        the model, loaded and moved to device in eval mode.
    """
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
    """
    Checks whether a small max_batches sample is skewed toward one label,
    since the paired loader is unshuffled and a small sample could
    happen to land on an unrepresentative slice of the data.

    Args:
        loader: paired loader.
        max_batches: how many batches to check.
        target_class: the attack's target class.

    Returns:
        dict with n_seen, label_counts, and frac_already_target.
    """
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
                             max_batches=None, save_dir=GRAPH_DIR):
    """
    Rules out generic importance as an explanation for a component's high
    cosine similarity with r_hat. Re-runs the same extraction on a clean,
    non-backdoored model, for the same components and the same r_hat, and
    compares. A component whose alignment is just as high on the clean
    model is reacting generically to the raw trigger patch as an image
    perturbation, not to the backdoor specifically.

    Args:
        clean_model: a clean, non-backdoored model of the same architecture.
        loader: paired loader.
        top_components: the components to check.
        r_hat: the globally chosen direction from phase0.
        target_class: the attack's target class.
        device: torch device.
        cls_position: which token position to compare.
        max_batches: optional cap.
        save_dir: where to save the plot.

    Returns:
        dict with the per-component comparison.
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
    return {"cosine_comparison": comparison}


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
    axes[0].set_ylabel("Cosine similarity of r^c with r_hat")
    axes[0].set_title("Backdoored vs clean model, each component's own r^c compared to r_hat")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, diff_vals, color=["crimson" if v > 0 else "steelblue" for v in diff_vals])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Difference, backdoored minus clean")
    axes[1].set_title("Backdoor specificity gap per component")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(_numbered_path(save_dir, "clean_model_control"), dpi=150); plt.close()


def run_phase1_pipeline(model, analysis_loader, directions_cls, r_hat, target_class, device,
                         top_k=10, max_batches_extract=None, additional_components=None,
                         run_logit_lens_flag=True,
                         run_all_heads_steering_sweep=True, steering_alphas=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
                         steering_sweep_max_batches=None, steering_sweep_dist_position=None,
                         cumulative_steering_alpha=1.0,
                         cumulative_steering_max_batches=None, cumulative_steering_thresholds=None,
                         clean_model=None, clean_model_max_batches=None, save_dir=None):
    """
    Runs the full Phase 1 pipeline: sanity check, component extraction
    and cosine ranking, direction consistency and alignment analysis,
    logit lens, cumulative cosine-ranked steering as the main causal
    validation step, and an optional whole-model steering sweep and
    clean-model control.

    Args:
        model: the backdoored model.
        analysis_loader: paired loader.
        directions_cls: dict layer -> r^l from phase0.
        r_hat: the globally chosen direction from phase0.
        target_class: the attack's target class.
        device: torch device.
        top_k: how many top-ranked components to test.
        max_batches_extract: cap for extraction and logit lens passes.
        additional_components: list of (layer, component) tuples force
            included regardless of cosine rank.
        run_logit_lens_flag: whether to run the logit lens.
        run_all_heads_steering_sweep: whether to run the expensive breadth sweep.
        steering_alphas: magnitudes for the breadth sweep.
        steering_sweep_max_batches: cap for the breadth sweep.
        steering_sweep_dist_position: token position of the DIST token,
            for DeiT-Tiny. Adds a DIST-only mode to the breadth sweep
            when given, extra cost, leave None for ViT-Small or to skip.
        cumulative_steering_alpha: magnitude for the cumulative steering test.
        cumulative_steering_max_batches: cap for the cumulative steering test.
            Now runs both CLS-only and all-tokens automatically, roughly
            doubling this step's cost versus a single-mode run.
        clean_model: optional clean model for the specificity control.
        clean_model_max_batches: cap for the clean-model control.
        save_dir: where every plot from this run is written. Pass
            get_graph_dir(model_name, attack_type, dataset) to keep each
            run's plots in their own folder.

    Returns:
        dict with top_components, all_scored_components, direction
        alignment results, logit lens results, cumulative steering
        results, breadth sweep results, clean model control results,
        and the extracted head and MLP directions.
    """
    save_dir = Path(save_dir) if save_dir is not None else GRAPH_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    if not sanity_check_manual_attention(model, analysis_loader, device):
        raise RuntimeError("Manual attention reconstruction does not match the real forward pass, fix before proceeding.")

    head_dirs, mlp_dirs = extract_component_activations(model, analysis_loader, device, max_batches=max_batches_extract)
    verify_extraction_consistency(head_dirs, mlp_dirs, directions_cls)
    top_components, all_scored = rank_components_by_cosine_similarity(head_dirs, mlp_dirs, r_hat, top_k=top_k)

    if additional_components:
        all_scored_lookup = {(d["layer"], d["component"]): d for d in all_scored}
        existing_keys = {(d["layer"], d["component"]) for d in top_components}
        extra = [all_scored_lookup[(l, c)] for (l, c) in additional_components
                 if (l, c) in all_scored_lookup and (l, c) not in existing_keys]
        top_components = top_components + extra

    plot_cosine_similarity_full_grid(all_scored, top_k=top_k, save_dir=save_dir)
    plot_direction_consistency_matrix(directions_cls, save_dir=save_dir)
    direction_alignment = compute_direction_alignment_with_rhat(directions_cls, r_hat, save_dir=save_dir)

    logit_lens_results = None
    if run_logit_lens_flag:
        logit_lens_results = logit_lens(model, analysis_loader, target_class, device, max_batches=max_batches_extract, save_dir=save_dir)

    cumulative_steering_results = cumulative_component_steering(
        model, analysis_loader, all_scored, head_dirs, mlp_dirs, target_class, device,
        alpha=cumulative_steering_alpha, max_batches=cumulative_steering_max_batches,
        thresholds=cumulative_steering_thresholds, save_dir=save_dir)

    steering_sweep_results = None
    if run_all_heads_steering_sweep:
        steering_sweep_results = sweep_all_heads_steering(model, analysis_loader, head_dirs, mlp_dirs, target_class, device,
                                                            alphas=steering_alphas, max_batches=steering_sweep_max_batches or max_batches_extract,
                                                            dist_position=steering_sweep_dist_position, save_dir=save_dir)

    clean_control_results = None
    if clean_model is not None:
        clean_control_results = run_clean_model_control(clean_model, analysis_loader, top_components, r_hat, target_class, device,
                                                          max_batches=clean_model_max_batches or max_batches_extract, save_dir=save_dir)

    return {
        "top_components": top_components, "all_scored_components": all_scored,
        "direction_alignment_with_rhat": direction_alignment,
        "logit_lens": logit_lens_results,
        "cumulative_component_steering": cumulative_steering_results,
        "steering_sweep_results": steering_sweep_results,
        "clean_model_control": clean_control_results,
        "head_dirs": head_dirs, "mlp_dirs": mlp_dirs,
    }