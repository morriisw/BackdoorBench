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

GRAPH_DIR  = Path("./results/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_STEER = 1.0   # steering scale applied to raw (unnormalised) direction

def _register_block_hooks(model):
    """
    Attach a forward hook to every TransformerBlock; overwritten each forward pass.
    """
    acts, hooks = {}, []
    for i, block in enumerate(model.transformer):
        def _hook(mod, inp, out, idx=i):
            acts[idx] = out.detach().cpu()   # (B, N_tokens, dim) — post-block residual stream
        hooks.append(block.register_forward_hook(_hook))
    return acts, hooks

@torch.no_grad()
def extract_paired_activations(model, loader, device, max_batches=None):
    """
    One pass over paired (clean, backdoored) batches, recording block-level activations for both. 
    This produces the X_pair = {(x, x_t)} used in the equation r^l = (1/|X_pair|) Σ (x_t^l − x^l).

    Returns
    -------
    clean_acts : dict[int (layer no.) -> Tensor (N, N_tokens, dim) = Tensor(images, tokens per image, embedding dim)]
    bd_acts    : dict[int (layer no.) -> Tensor (N, N_tokens, dim) = Tensor(images, tokens per image, embedding dim)]
    clean_labels, bd_labels : Tensor (N,)
    """
    model.eval()  # Disable dropout so activations are deterministic — required for a clean r^l estimate
    buf, hooks = _register_block_hooks(model)
    c_store, b_store = {}, {}
    c_lbls, b_lbls = [], []

    for i, (img_c, lbl_c, img_b, lbl_b) in enumerate(tqdm(loader, desc="Extracting activations")):
        if max_batches and i >= max_batches:
            break

        # Clean forward pass
        _ = model(img_c.to(device))
        for l, a in buf.items():
            c_store.setdefault(l, []).append(a)

        # Backdoored forward pass
        _ = model(img_b.to(device))
        for l, a in buf.items():
            b_store.setdefault(l, []).append(a)

        c_lbls.append(lbl_c); b_lbls.append(lbl_b)

    for h in hooks:
        h.remove()  # Detach hooks after use to avoid leaking them onto later forward passes

    clean_acts = {l: torch.cat(v) for l, v in c_store.items()}
    bd_acts = {l: torch.cat(v) for l, v in b_store.items()}
    return clean_acts, bd_acts, torch.cat(c_lbls), torch.cat(b_lbls)

def compute_backdoor_directions(clean_acts, bd_acts):
    """
    r^l = (1/N) Σ (x_t^l − x^l) over all pairs, for every layer l.

    Returns:
    directions_cls: dict[l -> (dim,)]:
        Used for BOTH layer selection and orthogonalization, since it's the only variant shape-compatible 
        with the weight matrices (dim, dim).

    directions_all: dict[l -> (N_tokens*dim,)]:
        Used ONLY for the steering analysis plots; cannot be used for orthogonalization as dimensions are 
        incompatible with the weight matrices.
    """
    directions_cls, directions_all = {}, {}
    for l in clean_acts:
        diff = bd_acts[l] - clean_acts[l]  # (N, N_tokens, d) = x_t^l − x^l
        directions_cls[l] = diff[:, 0, :].mean(0)  # CLS token only: (d,)
        directions_all[l] = diff.reshape(diff.size(0), -1).mean(0)  # all tokens concatenated: (N_tokens*d,)
    return directions_cls, directions_all

def plot_pca_layers(clean_acts, bd_acts, directions_cls,
                    layers_to_plot=(2, 5, 8, 11), tag="", save_dir=GRAPH_DIR):
    """
    Scatter plot (PC1 vs PC2) of CLS-token activations, clean (blue) vs
    backdoored (red), with the backdoor direction projected as an arrow.
    Matches Karayalcin Figure 1 style.
    """
    layers_to_plot = [l for l in layers_to_plot if l in clean_acts]
    n = len(layers_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, l in zip(axes, layers_to_plot):
        c_cls = clean_acts[l][:, 0, :].numpy()
        b_cls = bd_acts[l][:,   0, :].numpy()
        N = len(c_cls)

        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(np.concatenate([c_cls, b_cls]))
        c2, b2 = all_2d[:N], all_2d[N:]

        ax.scatter(c2[:, 0], c2[:, 1], c="blue", alpha=0.25, s=6)
        ax.scatter(b2[:, 0], b2[:, 1], c="red", alpha=0.25, s=6)

        # Project backdoor direction onto PC space
        d = directions_cls[l].numpy()            # (dim,)
        d2 = pca.transform(d.reshape(1, -1))[0]  # (2,)

        # Scale arrow relative to data spread
        scale = np.std(all_2d) * 1.5
        d_unit = d2 / (np.linalg.norm(d2) + 1e-8) * scale
        ax.annotate("", xy=d_unit, xytext=(0., 0.),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=2.0, mutation_scale=15))

        ax.set_title(f"Layer {l + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=9); ax.set_ylabel("PC2", fontsize=9)
        ax.tick_params(labelsize=7)

    handles = [mpatches.Patch(color="blue", label="Clean"),
               mpatches.Patch(color="red", label="Backdoored"),
               plt.Line2D([0], [0], color="black", lw=2, label="BD direction")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PCA of CLS Token Activations: Clean vs Backdoored",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = save_dir / f"pca_activations{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def _eval_steered(model, loader, layer_idx, direction, mode,
                  target_class, device, max_batches=None,
                  steer_all_tokens=False, n_tokens=None):
    """
    mode='positive': add r^l to CLEAN inputs at layer_idx, check if prediction
                      flips to target_class -> gives ASR_{+r^l}.
    mode='negative': subtract r^l from BACKDOORED inputs at layer_idx, check if
                      prediction is restored to the original clean label -> gives RA_{-r^l}.
    """
    model.eval()
    d = direction.to(device)
    if steer_all_tokens and n_tokens is not None:
        d_per_token = d.view(n_tokens, -1)  # Reshape concatenated direction back to (N_tokens, d)

    def _hook(mod, inp, out):
        steered = out.clone()
        if steer_all_tokens:
            delta = ALPHA_STEER * d_per_token.unsqueeze(0)
            steered = steered + delta if mode == "positive" else steered - delta
        else:
            sign = 1 if mode == "positive" else -1
            steered[:, 0, :] += sign * ALPHA_STEER * d  # Only the CLS token position is steered
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
            correct += (preds == target_class).sum().item()  # ASR_{+r^l}: flipped to target class
        else:
            correct += (preds == lbl_c).sum().item()  # RA_{-r^l}: restored to original clean label
        total += len(img_c)
    hook.remove()
    return correct / total if total else 0.0

def run_layer_wise_steering(model, loader, directions_cls, directions_all,
                             target_class, device, max_batches=None, n_tokens=None):
    """
    Runs positive/negative steering at every layer, for both CLS and all-token directions.
    """
    n_layers = len(directions_cls)
    results = {v: {"asr_pos": [], "ra_neg": []} for v in ("cls", "all")}
    for l in tqdm(range(n_layers), desc="Layer-wise steering"):
        for key, dirs, all_tok in [("cls", directions_cls, False), ("all", directions_all, True)]:
            asr_pos = _eval_steered(model, loader, l, dirs[l], "positive",
                                    target_class, device, max_batches, all_tok, n_tokens)
            ra_neg  = _eval_steered(model, loader, l, dirs[l], "negative",
                                    target_class, device, max_batches, all_tok, n_tokens)
            results[key]["asr_pos"].append(asr_pos)
            results[key]["ra_neg"].append(ra_neg)
    return results

def select_most_representative_layer(steering_results):
    """
    r_hat = argmax_l { (ASR_{+r^l} + RA_{-r^l}) - (ASR_{+r^{l-1}} + RA_{-r^{l-1}}) }
    l_hat is the layer whose direction produces the largest DELTA (jump) in
    combined score vs. the previous layer.
    """
    asr_pos = np.array(steering_results["cls"]["asr_pos"])
    ra_neg = np.array(steering_results["cls"]["ra_neg"])
    combined = asr_pos + ra_neg
    deltas = combined[1:] - combined[:-1]  # Eq. 1
    l_hat = int(np.argmax(deltas)) + 1  # +1 because deltas[i] corresponds to layer i+1
    return l_hat, combined, deltas

def global_orthogonalize(model, direction, device):
    """
    Remove the backdoor direction from every matrix that writes into the residual stream. 
    Only nn.Linear layers qualify: LayerNorm weights are elementwise gains, not writing 
    matrices, and should not be orthogonalized.
    """
    model = deepcopy(model)
    r = direction.to(device)
    r = r / (r.norm() + 1e-8)  # Must be unit norm for P to be a true projector
    P = torch.eye(r.size(0), device=device) - torch.outer(r, r)

    with torch.no_grad():
        # Initial embedding layer: only the Linear that projects flattened patches into 
        # dim space writes into the residual stream
        for name, m in model.named_modules():
            if "patch_embed" in name and isinstance(m, nn.Linear) and m.weight.size(0) == r.size(0):
                m.weight.data = P @ m.weight.data
                if m.bias is not None:
                    m.bias.data = P @ m.bias.data

        # Every transformer block's attention output projection and MLP output projection
        for blk in model.transformer:
            blk.attn.out_proj.weight.data = P @ blk.attn.out_proj.weight.data
            if blk.attn.out_proj.bias is not None:
                blk.attn.out_proj.bias.data = P @ blk.attn.out_proj.bias.data

            lin_layers = [m for m in blk.mlp.modules() if isinstance(m, nn.Linear)]
            if lin_layers:
                lin_layers[-1].weight.data = P @ lin_layers[-1].weight.data  # Last linear = MLP's output projection
                if lin_layers[-1].bias is not None:
                    lin_layers[-1].bias.data = P @ lin_layers[-1].bias.data

    return model

def plot_steering_results(steering_results, baseline, target_name, save_dir):
    """
    Single plot showing:
      - ASR+ and RA- for All-token steering
      - ASR+ and RA- for CLS-token steering
      - Baseline Clean Accuracy, ASR, and RA
    """
    n = len(steering_results["cls"]["asr_pos"])
    layers = list(range(n))
    s = steering_results

    plt.figure(figsize=(10, 6))

    # All-token steering
    plt.plot(layers, s["all"]["asr_pos"], "r-o", lw=2, label="ASR+ (All Tokens)",)
    plt.plot(layers,s["all"]["ra_neg"],"b-s",lw=2,label="RA- (All Tokens)",)

    # CLS-token steering
    plt.plot(layers,s["cls"]["asr_pos"],"r--o",lw=2,alpha=0.7,label="ASR+ (CLS Token)",)
    plt.plot(layers,s["cls"]["ra_neg"],"b--s",lw=2,alpha=0.7,label="RA- (CLS Token)",)

    # Baselines
    plt.axhline(baseline[0],color="gray",linestyle=":",linewidth=2,label=f"Baseline Clean Acc = {baseline[0]:.3f}",)
    plt.axhline(baseline[1],color="red",linestyle="--",alpha=0.6,label=f"Baseline ASR = {baseline[1]:.3f}",)
    plt.axhline(baseline[2],color="green",linestyle="--",alpha=0.6,label=f"Baseline RA = {baseline[2]:.3f}",)

    plt.xlabel("Layer")
    plt.ylabel("Rate")
    plt.title(f"Steering Results ({target_name})")
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    path = save_dir / "steering_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def run_phase0_pipeline(model, analysis_loader, target_class, device="cuda", 
                        max_batches_extract=None, max_batches_steer=None):
    clean_acts, bd_acts, _, _ = extract_paired_activations(model, analysis_loader, device, max_batches_extract)
    directions_cls, directions_all = compute_backdoor_directions(clean_acts, bd_acts)
    plot_pca_layers(clean_acts, bd_acts, directions_cls)

    # Before weight orthogonalisation defence
    baseline = evaluate_acc_asr(analysis_loader, model, device)
    print(f"\nBaseline - ACC: {baseline[0]:.4f}, ASR: {baseline[1]:.4f}, RA: {baseline[2]:.4f}")

    # Apply steering and identify most representative layer (l*)
    # Read sequence length directly from the model's learned position embedding
    n_tokens = model.pos_embedding.shape[1]

    steering_results = run_layer_wise_steering(
        model, analysis_loader, directions_cls, directions_all,
        target_class, device, max_batches_steer, n_tokens=n_tokens)

    # Print layer-wise steering results
    n_layers = len(directions_cls)
    print(f"\n{'Layer':>5}  {'ASR+(CLS)':>9}  {'RA-(CLS)':>9}  {'ASR+(All)':>9}  {'RA-(All)':>9}")
    for l in range(n_layers):
        print(f"{l:>5}  {steering_results['cls']['asr_pos'][l]:>9.4f}  "
              f"{steering_results['cls']['ra_neg'][l]:>9.4f}  "
              f"{steering_results['all']['asr_pos'][l]:>9.4f}  "
              f"{steering_results['all']['ra_neg'][l]:>9.4f}")

    # Find most representative layer
    l_hat, cls_scores, all_scores = select_most_representative_layer(steering_results)
    print(f"\nMost Representative layer (l*): {l_hat}")
    print(f"  ASR+(CLS) at l* = {steering_results['cls']['asr_pos'][l_hat]:.4f}")
    print(f"  RA-(CLS) at l*  = {steering_results['cls']['ra_neg'][l_hat]:.4f}")

    # Remove backdoor direction from the most representative layer (l*) and evaluate metrics
    # Global orthogonalisation
    r_hat = directions_cls[l_hat]
    model_defended = global_orthogonalize(model, r_hat, device)
    final = evaluate_acc_asr(analysis_loader, model_defended, device)
    print(f"\nAfter global orthogonalisation - ACC: {final[0]:.4f}, ASR: {final[1]:.4f}, RA: {final[2]:.4f}")
    print(f"ASR reduction: {baseline[1]-final[1]:.4f} ({100*(baseline[1]-final[1])/baseline[1]:.1f}%)")

    # Steering plot
    plot_steering_results(steering_results, baseline, f"Target Class {target_class}", GRAPH_DIR)

    return {
        "model_defended": model_defended,       # globally-orthogonalised model (Phase 2 layer-level baseline)
        "directions_cls": directions_cls,       # dict[l] -> r^l (CLS-token backdoor direction, every layer)
        "directions_all": directions_all,       # dict[l] -> r^l (all-token variant, every layer)
        "l_hat": l_hat,                         # most representative layer index
        "r_hat": r_hat,                         # = directions_cls[l_hat], the single direction used for defence
        "steering_results": steering_results,   # per-layer ASR+/RA- curves (both cls and all variants)
        "baseline_metrics": baseline,           # (ACC, ASR, RA) BEFORE defence
        "defended_metrics": final,              # (ACC, ASR, RA) AFTER global orthogonalisation
        "n_tokens": n_tokens,
    }

