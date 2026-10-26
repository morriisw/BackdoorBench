"""
Phase 0: extracts the backdoor direction, finds the most representative
layer through steering, and applies whole-model weight orthogonalisation
as the layer-wide baseline defence.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from eval_general import evaluate_acc_asr

GRAPH_DIR = Path("./results/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_STEER = 1.0  # steering scale applied to the raw, unnormalised direction


def get_graph_dir(model_name, attack_type, dataset, base="results"):
    """
    Builds and returns results/{model_name}/{attack_type}/{dataset}.
    """
    path = Path(base) / model_name / attack_type / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def _register_block_hooks(model):
    """
    Attaches a forward hook to every transformer block.

    Returns:
        acts: dict layer -> tensor (B, N_tokens, dim), post-block residual stream.
        hooks: list of hook handles, remove these when done.
    """
    acts, hooks = {}, []
    for i, block in enumerate(model.transformer):
        def _hook(mod, inp, out, idx=i):
            acts[idx] = out.detach().cpu()
        hooks.append(block.register_forward_hook(_hook))
    return acts, hooks


@torch.no_grad()
def extract_paired_activations(model, loader, device, max_batches=None):
    """
    One pass over paired clean and backdoored batches, recording
    block-level activations for both. This produces the paired activation
    set used to compute r^l = mean(x_triggered^l - x_clean^l).

    Args:
        model: the ViT model.
        loader: paired loader.
        device: torch device.
        max_batches: optional cap on how many batches to process.

    Returns:
        clean_acts: dict layer -> tensor (N, N_tokens, dim).
        bd_acts: dict layer -> tensor (N, N_tokens, dim).
        clean_labels, bd_labels: tensor (N,).
    """
    model.eval()
    buf, hooks = _register_block_hooks(model)
    c_store, b_store = {}, {}
    c_lbls, b_lbls = [], []

    for i, (img_c, lbl_c, img_b, lbl_b) in enumerate(tqdm(loader, desc="Extracting activations")):
        if max_batches and i >= max_batches:
            break

        _ = model(img_c.to(device))
        for l, a in buf.items():
            c_store.setdefault(l, []).append(a)

        _ = model(img_b.to(device))
        for l, a in buf.items():
            b_store.setdefault(l, []).append(a)

        c_lbls.append(lbl_c); b_lbls.append(lbl_b)

    for h in hooks:
        h.remove()

    clean_acts = {l: torch.cat(v) for l, v in c_store.items()}
    bd_acts = {l: torch.cat(v) for l, v in b_store.items()}
    return clean_acts, bd_acts, torch.cat(c_lbls), torch.cat(b_lbls)


def compute_backdoor_directions(clean_acts, bd_acts, dist_position=None):
    """
    Computes r^l = mean(x_triggered^l - x_clean^l) for every layer.

    Args:
        clean_acts, bd_acts: from extract_paired_activations.
        dist_position: token position of the DIST token for DeiT-Tiny.
            Leave None for models without a distillation token.

    Returns:
        directions_cls: dict layer -> vector (dim,), CLS position only.
            Used for both layer selection and orthogonalisation, since
            this is the only shape compatible with the weight matrices.
        directions_all: dict layer -> vector (N_tokens*dim,), every token
            concatenated. Used only for the all-tokens steering mode,
            cannot be used for orthogonalisation.
        directions_dist: dict layer -> vector (dim,) at dist_position, or
            empty dict if dist_position is None. Kept separate from
            directions_cls rather than assumed identical, since CLS and
            DIST are trained by different loss terms and may encode the
            backdoor differently.
    """
    directions_cls, directions_all, directions_dist = {}, {}, {}
    for l in clean_acts:
        diff = bd_acts[l] - clean_acts[l]
        directions_cls[l] = diff[:, 0, :].mean(0)
        directions_all[l] = diff.reshape(diff.size(0), -1).mean(0)
        if dist_position is not None:
            directions_dist[l] = diff[:, dist_position, :].mean(0)
    return directions_cls, directions_all, directions_dist


def compute_cls_dist_alignment(directions_cls, directions_dist, save_dir=GRAPH_DIR):
    """
    Plots cosine similarity between the CLS-token and DIST-token backdoor
    directions, per layer. High similarity means CLS and DIST are moving
    the same way, consistent with DIST largely duplicating CLS's signal.
    Low or negative similarity means DIST is doing something different at
    that layer.

    Args:
        directions_cls, directions_dist: dict layer -> vector.
        save_dir: where to save the plot.

    Returns:
        dict layer -> cosine similarity.
    """
    import torch.nn.functional as F
    layers = sorted(directions_cls.keys())
    sims = [F.cosine_similarity(directions_cls[l].unsqueeze(0), directions_dist[l].unsqueeze(0)).item() for l in layers]

    plt.figure(figsize=(9, 5))
    plt.bar([str(l) for l in layers], sims, color=["crimson" if s > 0 else "steelblue" for s in sims])
    plt.axhline(0, color="k", lw=0.8)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Layer"); plt.ylabel("Cosine similarity, r^l CLS vs r^l DIST")
    plt.title("CLS vs DIST backdoor direction alignment, by layer")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "cls_dist_alignment.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")
    return {l: s for l, s in zip(layers, sims)}


def plot_pca_layers(clean_acts, bd_acts, directions_cls, layers_to_plot=(2, 5, 8, 11), tag="", save_dir=GRAPH_DIR):
    """
    Plots a PCA scatter of CLS-token activations, clean versus
    backdoored, with the backdoor direction projected as an arrow.

    Args:
        clean_acts, bd_acts: from extract_paired_activations.
        directions_cls: dict layer -> r^l.
        layers_to_plot: which layers to show.
        tag: appended to the output filename.
        save_dir: where to save the plot.
    """
    layers_to_plot = [l for l in layers_to_plot if l in clean_acts]
    n = len(layers_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, l in zip(axes, layers_to_plot):
        c_cls = clean_acts[l][:, 0, :].numpy()
        b_cls = bd_acts[l][:, 0, :].numpy()
        N = len(c_cls)

        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(np.concatenate([c_cls, b_cls]))
        c2, b2 = all_2d[:N], all_2d[N:]

        ax.scatter(c2[:, 0], c2[:, 1], c="blue", alpha=0.25, s=6)
        ax.scatter(b2[:, 0], b2[:, 1], c="red", alpha=0.25, s=6)

        d = directions_cls[l].numpy()
        d2 = pca.transform(d.reshape(1, -1))[0]

        scale = np.std(all_2d) * 1.5
        d_unit = d2 / (np.linalg.norm(d2) + 1e-8) * scale
        ax.annotate("", xy=d_unit, xytext=(0., 0.),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0, mutation_scale=15))

        ax.set_title(f"Layer {l}", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=9); ax.set_ylabel("PC2", fontsize=9)
        ax.tick_params(labelsize=7)

    handles = [mpatches.Patch(color="blue", label="Clean"),
               mpatches.Patch(color="red", label="Backdoored"),
               plt.Line2D([0], [0], color="black", lw=2, label="Backdoor direction")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PCA of CLS token activations, clean vs backdoored", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"pca_activations{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def _eval_steered(model, loader, layer_idx, direction, mode, target_class, device,
                   max_batches=None, steer_all_tokens=False, n_tokens=None, token_position=0):
    """
    Steers one layer's output by direction and measures the result.

    Args:
        model: the ViT model.
        loader: paired loader.
        layer_idx: which layer to steer.
        direction: the vector to steer by.
        mode: "positive" adds r^l to clean inputs and checks ASR.
            "negative" subtracts r^l from backdoored inputs and checks RA.
        target_class: the attack's target class.
        device: torch device.
        max_batches: optional cap.
        steer_all_tokens: if True, steers every token position using
            direction reshaped as (N_tokens, dim).
        n_tokens: required when steer_all_tokens is True.
        token_position: which single token to steer when steer_all_tokens
            is False. 0 is CLS, 1 is DIST for DeiT-Tiny.

    Returns:
        ASR (positive mode) or RA (negative mode) as a fraction.
    """
    model.eval()
    d = direction.to(device)
    if steer_all_tokens and n_tokens is not None:
        d_per_token = d.view(n_tokens, -1)

    def _hook(mod, inp, out):
        steered = out.clone()
        if steer_all_tokens:
            delta = ALPHA_STEER * d_per_token.unsqueeze(0)
            steered = steered + delta if mode == "positive" else steered - delta
        else:
            sign = 1 if mode == "positive" else -1
            steered[:, token_position, :] += sign * ALPHA_STEER * d
        return steered

    hook = model.transformer[layer_idx].register_forward_hook(_hook)
    correct = total = 0
    for i, (img_c, lbl_c, img_b, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        inputs = img_c.to(device) if mode == "positive" else img_b.to(device)
        out = model(inputs)
        preds = out.argmax(1).cpu()
        if mode == "positive":
            correct += (preds == target_class).sum().item()
        else:
            correct += (preds == lbl_c).sum().item()
        total += len(img_c)
    hook.remove()
    return correct / total if total else 0.0


def _eval_steered_joint_positions(model, loader, layer_idx, position_directions, mode, target_class, device, max_batches=None):
    """
    Steers several token positions at once, each with its own direction,
    leaving every other position untouched. Different from steer_all_tokens
    in _eval_steered, which touches every position including all patch
    tokens, this touches only the given positions.

    Args:
        model: the ViT model.
        loader: paired loader.
        layer_idx: which layer to steer.
        position_directions: list of (token_position, direction) pairs.
        mode: "positive" or "negative", same meaning as _eval_steered.
        target_class: the attack's target class.
        device: torch device.
        max_batches: optional cap.

    Returns:
        ASR (positive mode) or RA (negative mode) as a fraction.
    """
    model.eval()
    sign = 1 if mode == "positive" else -1

    def _hook(mod, inp, out):
        steered = out.clone()
        for pos, direction in position_directions:
            steered[:, pos, :] += sign * ALPHA_STEER * direction.to(device)
        return steered

    hook = model.transformer[layer_idx].register_forward_hook(_hook)
    correct = total = 0
    for i, (img_c, lbl_c, img_b, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        inputs = img_c.to(device) if mode == "positive" else img_b.to(device)
        preds = model(inputs).argmax(1).cpu()
        if mode == "positive":
            correct += (preds == target_class).sum().item()
        else:
            correct += (preds == lbl_c).sum().item()
        total += len(img_c)
    hook.remove()
    return correct / total if total else 0.0


def run_layer_wise_steering(model, loader, directions_cls, directions_all, target_class, device,
                             max_batches=None, n_tokens=None, directions_dist=None, dist_position=1):
    """
    Runs positive and negative steering at every layer, for CLS-token and
    all-token directions, and for DIST-token directions too when
    directions_dist is given. DIST is tested as its own isolated
    position, not folded into all-tokens mode, since the point is
    checking whether steering DIST alone flips predictions the same way
    CLS-mode already checks for CLS.

    Also runs a joint mode when directions_dist is given: steers only
    CLS and DIST simultaneously, each with its own direction, leaving
    every patch token untouched. This isolates whether the two special
    tokens together are jointly sufficient, without the confound of
    all-tokens mode also steering every patch token at the same time.

    Args:
        model: the ViT model.
        loader: paired loader.
        directions_cls, directions_all: from compute_backdoor_directions.
        target_class: the attack's target class.
        device: torch device.
        max_batches: optional cap per evaluation.
        n_tokens: total token count, needed for all-tokens mode.
        directions_dist: dict layer -> DIST direction, or None to skip DIST tests.
        dist_position: token position of the DIST token.

    Returns:
        dict with cls, all, and if applicable dist and cls_dist_joint
        results, each holding per-layer asr_pos and ra_neg lists.
    """
    n_layers = len(directions_cls)
    modes = [("cls", directions_cls, False, 0)]
    if directions_dist:
        modes.append(("dist", directions_dist, False, dist_position))
    modes.append(("all", directions_all, True, 0))

    results = {key: {"asr_pos": [], "ra_neg": []} for key, _, _, _ in modes}
    if directions_dist:
        results["cls_dist_joint"] = {"asr_pos": [], "ra_neg": []}

    for l in tqdm(range(n_layers), desc="Layer-wise steering"):
        for key, dirs, all_tok, pos in modes:
            asr_pos = _eval_steered(model, loader, l, dirs[l], "positive", target_class, device, max_batches, all_tok, n_tokens, pos)
            ra_neg = _eval_steered(model, loader, l, dirs[l], "negative", target_class, device, max_batches, all_tok, n_tokens, pos)
            results[key]["asr_pos"].append(asr_pos)
            results[key]["ra_neg"].append(ra_neg)

        if directions_dist:
            pairs = [(0, directions_cls[l]), (dist_position, directions_dist[l])]
            asr_pos = _eval_steered_joint_positions(model, loader, l, pairs, "positive", target_class, device, max_batches)
            ra_neg = _eval_steered_joint_positions(model, loader, l, pairs, "negative", target_class, device, max_batches)
            results["cls_dist_joint"]["asr_pos"].append(asr_pos)
            results["cls_dist_joint"]["ra_neg"].append(ra_neg)

    return results


def select_most_representative_layer(steering_results, mode="cls"):
    """
    Finds the layer whose direction produces the largest jump in combined
    ASR plus RA score versus the previous layer.

    Args:
        steering_results: from run_layer_wise_steering.
        mode: which key to select from, "cls" or "dist". Each mode gets
            its own independent selection, so a DIST-vs-CLS comparison
            is not confounded by DIST always being scored at whichever
            layer happened to be best for CLS.

    Returns:
        l_hat: the selected layer index.
        combined: the combined ASR plus RA score per layer.
        deltas: the jump at each layer versus the previous one.
    """
    asr_pos = np.array(steering_results[mode]["asr_pos"])
    ra_neg = np.array(steering_results[mode]["ra_neg"])
    combined = asr_pos + ra_neg
    deltas = combined[1:] - combined[:-1]
    l_hat = int(np.argmax(deltas)) + 1
    return l_hat, combined, deltas


def _build_projector(directions, device):
    """
    Builds the projection matrix that removes the subspace spanned by one
    or more directions. For a single direction this is the usual
    P = I - r_hat r_hat^T. For several directions, an orthonormal basis
    of their span is built first via QR decomposition, so the whole
    subspace is removed at once rather than each direction's own 1D
    projector applied in sequence, which can partially reintroduce a
    previously removed direction if the two are not already orthogonal.

    Args:
        directions: list of direction vectors.
        device: torch device.

    Returns:
        the projection matrix P.
    """
    stacked = torch.stack([d.to(device) for d in directions], dim=1)
    Q, _ = torch.linalg.qr(stacked, mode="reduced")
    return torch.eye(Q.size(0), device=device) - Q @ Q.T


def global_orthogonalize(model, direction, device):
    """
    Removes direction from every matrix that writes into the residual
    stream: the patch embedding, and every layer's attention output
    projection and MLP output projection. This is the whole-model,
    layer-wide baseline defence.

    Args:
        model: the model to defend, not modified in place.
        direction: the direction to remove.
        device: torch device.

    Returns:
        a deep copy of model with direction removed everywhere it writes.
    """
    return _global_orthogonalize_with_projector(model, _build_projector([direction], device), device)


def global_orthogonalize_multi(model, directions, device):
    """
    Same operation as global_orthogonalize, but removes the joint
    subspace spanned by several directions at once, rather than picking
    one direction alone. Meaningful because weight-level orthogonalisation
    is inherently all-position, there is no way to instead apply each
    direction at its own token position, so a both-together test
    here means a joint subspace.

    Args:
        model: the model to defend, not modified in place.
        directions: list of directions to remove together.
        device: torch device.

    Returns:
        a deep copy of model with the joint subspace removed.
    """
    return _global_orthogonalize_with_projector(model, _build_projector(directions, device), device)


def global_orthogonalize_attn_only(model, direction, device):
    """
    Removes direction from every layer's attention output projection
    only, leaving the patch embedding and every MLP untouched. Unlike 
    global_orthogonalize, this does not project out_proj.bias.
 
    Args:
        model: the model to defend, not modified in place.
        direction: the direction to remove.
        device: torch device.
 
    Returns:
        a deep copy of model with direction removed from every
        attention output projection's weight only.
    """
    model = deepcopy(model)
    P = _build_projector([direction], device)
    with torch.no_grad():
        for blk in model.transformer:
            blk.attn.out_proj.weight.data = P @ blk.attn.out_proj.weight.data
    return model


def _global_orthogonalize_with_projector(model, P, device):
    model = deepcopy(model)
    with torch.no_grad():
        for name, m in model.named_modules():
            if "patch_embed" in name and isinstance(m, nn.Linear) and m.weight.size(0) == P.size(0):
                m.weight.data = P @ m.weight.data
                if m.bias is not None:
                    m.bias.data = P @ m.bias.data

        for blk in model.transformer:
            blk.attn.out_proj.weight.data = P @ blk.attn.out_proj.weight.data
            if blk.attn.out_proj.bias is not None:
                blk.attn.out_proj.bias.data = P @ blk.attn.out_proj.bias.data

            lin_layers = [m for m in blk.mlp.modules() if isinstance(m, nn.Linear)]
            if lin_layers:
                lin_layers[-1].weight.data = P @ lin_layers[-1].weight.data
                if lin_layers[-1].bias is not None:
                    lin_layers[-1].bias.data = P @ lin_layers[-1].bias.data

    return model


def plot_steering_results(steering_results, baseline, target_name, save_dir=GRAPH_DIR):
    """
    Plots ASR and RA for all-token and CLS-token steering, plus DIST and
    joint steering if present, alongside the unsteered baseline.

    Args:
        steering_results: from run_layer_wise_steering.
        baseline: (acc, asr, ra) tuple, unsteered.
        target_name: label for the plot title.
        save_dir: where to save the plot.
    """
    n = len(steering_results["cls"]["asr_pos"])
    layers = list(range(n))
    s = steering_results

    plt.figure(figsize=(10, 6))

    plt.plot(layers, s["all"]["asr_pos"], "r-o", lw=2, label="ASR, all tokens")
    plt.plot(layers, s["all"]["ra_neg"], "b-s", lw=2, label="RA, all tokens")

    plt.plot(layers, s["cls"]["asr_pos"], "r--o", lw=2, alpha=0.7, label="ASR, CLS token")
    plt.plot(layers, s["cls"]["ra_neg"], "b--s", lw=2, alpha=0.7, label="RA, CLS token")

    if "dist" in s:
        plt.plot(layers, s["dist"]["asr_pos"], "r:^", lw=2, alpha=0.9, label="ASR, DIST token")
        plt.plot(layers, s["dist"]["ra_neg"], "b:^", lw=2, alpha=0.9, label="RA, DIST token")

    if "cls_dist_joint" in s:
        plt.plot(layers, s["cls_dist_joint"]["asr_pos"], "r-.d", lw=2, alpha=0.9, label="ASR, CLS+DIST joint")
        plt.plot(layers, s["cls_dist_joint"]["ra_neg"], "b-.d", lw=2, alpha=0.9, label="RA, CLS+DIST joint")

    plt.axhline(baseline[0], color="gray", linestyle=":", linewidth=2, label=f"Baseline clean ACC = {baseline[0]:.3f}")
    plt.axhline(baseline[1], color="red", linestyle="--", alpha=0.6, label=f"Baseline ASR = {baseline[1]:.3f}")
    plt.axhline(baseline[2], color="green", linestyle="--", alpha=0.6, label=f"Baseline RA = {baseline[2]:.3f}")

    plt.xlabel("Layer")
    plt.ylabel("Rate")
    plt.title(f"Steering results, {target_name}")
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "steering_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def run_phase0_pipeline(model, analysis_loader, target_class, device="cuda",
                         max_batches_extract=None, max_batches_steer=None, save_dir=None, dist_position=None):
    """
    Runs the full Phase 0 pipeline: extracts r^l for every layer, plots
    PCA scatters, runs layer-wise steering to find the most representative
    layer, and applies whole-model orthogonalisation as the baseline
    defence.

    Only the CLS-based direction and defence are computed automatically.
    DIST's own direction and joint subspace removal are cheap to run on
    demand using global_orthogonalize and global_orthogonalize_multi
    directly, evaluating a defended model belongs to Phase 2's comparison
    rather than being baked into every Phase 0 call.

    Args:
        model: the backdoored model.
        analysis_loader: paired loader.
        target_class: the attack's target class.
        device: torch device.
        max_batches_extract: cap for direction extraction.
        max_batches_steer: cap for steering evaluations.
        save_dir: where every plot from this run is written. Pass
            get_graph_dir(model_name, attack_type, dataset), imported
            from phase1, to keep each run's plots in their own folder.
        dist_position: set to 1 for DeiT-Tiny to also extract the
            DIST-token direction and run DIST-specific steering. Leave
            None for models without a distillation token, every
            CLS-based result stays exactly the same either way.

    Returns:
        dict with the defended model, all extracted directions, the
        selected layer, r_hat, steering results, baseline and defended
        metrics, and DIST-related directions if dist_position was given.
    """
    save_dir = Path(save_dir) if save_dir is not None else GRAPH_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    clean_acts, bd_acts, _, _ = extract_paired_activations(model, analysis_loader, device, max_batches_extract)
    directions_cls, directions_all, directions_dist = compute_backdoor_directions(clean_acts, bd_acts, dist_position=dist_position)
    plot_pca_layers(clean_acts, bd_acts, directions_cls, save_dir=save_dir)

    cls_dist_alignment = None
    if dist_position is not None:
        cls_dist_alignment = compute_cls_dist_alignment(directions_cls, directions_dist, save_dir=save_dir)

    baseline = evaluate_acc_asr(analysis_loader, model, device)
    print(f"\nBaseline - ACC: {baseline[0]:.4f}, ASR: {baseline[1]:.4f}, RA: {baseline[2]:.4f}")

    n_tokens = model.pos_embedding.shape[1]

    steering_results = run_layer_wise_steering(
        model, analysis_loader, directions_cls, directions_all,
        target_class, device, max_batches_steer, n_tokens=n_tokens,
        directions_dist=directions_dist, dist_position=dist_position if dist_position is not None else 1)

    n_layers = len(directions_cls)
    has_dist = "dist" in steering_results
    header = f"\n{'Layer':>5}  {'ASR CLS':>9}  {'RA CLS':>9}  {'ASR All':>9}  {'RA All':>9}"
    if has_dist:
        header += f"  {'ASR DIST':>10}  {'RA DIST':>9}  {'ASR Joint':>11}  {'RA Joint':>10}"
    print(header)
    for l in range(n_layers):
        row = (f"{l:>5}  {steering_results['cls']['asr_pos'][l]:>9.4f}  "
               f"{steering_results['cls']['ra_neg'][l]:>9.4f}  "
               f"{steering_results['all']['asr_pos'][l]:>9.4f}  "
               f"{steering_results['all']['ra_neg'][l]:>9.4f}")
        if has_dist:
            row += (f"  {steering_results['dist']['asr_pos'][l]:>10.4f}  "
                    f"{steering_results['dist']['ra_neg'][l]:>9.4f}  "
                    f"{steering_results['cls_dist_joint']['asr_pos'][l]:>11.4f}  "
                    f"{steering_results['cls_dist_joint']['ra_neg'][l]:>10.4f}")
        print(row)

    l_hat, cls_scores, all_scores = select_most_representative_layer(steering_results)
    print(f"\nMost representative layer: {l_hat}")
    print(f"  ASR CLS at l_hat = {steering_results['cls']['asr_pos'][l_hat]:.4f}")
    print(f"  RA CLS at l_hat  = {steering_results['cls']['ra_neg'][l_hat]:.4f}")

    r_hat = directions_cls[l_hat]
    model_defended = global_orthogonalize(model, r_hat, device)
    final = evaluate_acc_asr(analysis_loader, model_defended, device)
    print(f"\nAfter global orthogonalisation, CLS direction - ACC: {final[0]:.4f}, ASR: {final[1]:.4f}, RA: {final[2]:.4f}")
    print(f"ASR reduction: {baseline[1] - final[1]:.4f} ({100 * (baseline[1] - final[1]) / baseline[1]:.1f}%)")

    r_hat_dist = l_hat_dist = r_hat_dist_optimal = None
    if directions_dist:
        r_hat_dist = directions_dist[l_hat]
        l_hat_dist, _, _ = select_most_representative_layer(steering_results, mode="dist")
        r_hat_dist_optimal = directions_dist[l_hat_dist]
        print(f"\nDIST's own most representative layer: {l_hat_dist}")

    plot_steering_results(steering_results, baseline, f"Target class {target_class}", save_dir=save_dir)

    return {
        "model_defended": model_defended,
        "directions_cls": directions_cls,
        "directions_all": directions_all,
        "directions_dist": directions_dist,
        "cls_dist_alignment": cls_dist_alignment,
        "l_hat": l_hat,
        "r_hat": r_hat,
        "r_hat_dist": r_hat_dist,
        "l_hat_dist": l_hat_dist,
        "r_hat_dist_optimal": r_hat_dist_optimal,
        "steering_results": steering_results,
        "baseline_metrics": baseline,
        "defended_metrics": final,
        "n_tokens": n_tokens,
        "save_dir": save_dir,
    }