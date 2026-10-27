"""
Phase 2: component-level weight orthogonalisation.

Extends phase0's global_orthogonalize down to individual components
found in Phase 1, and compares narrow (component-level) against broad
(whole-model) removal on ACC, ASR, and RA.

write_only orthogonalises only the matrices that write a component's
own contribution into the residual stream. read_write additionally
orthogonalises the matrices that read into it, changing what the
component computes, not just what it writes out.
"""
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from phase0 import global_orthogonalize, global_orthogonalize_multi


def get_graph_dir(model_name, attack_type, dataset, base="results"):
    """Builds and returns results/{model_name}/{attack_type}/{dataset}."""
    path = Path(base) / model_name / attack_type / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def orthogonalize_head_write(attn_module, head_idx, P):
    """Projects a direction out of head_idx's column slice of out_proj.weight."""
    D, H = attn_module.embed_dim, attn_module.num_heads
    hd = D // H
    c0, c1 = head_idx * hd, (head_idx + 1) * hd
    with torch.no_grad():
        Wo = attn_module.out_proj.weight.data
        Wo[:, c0:c1] = P @ Wo[:, c0:c1]


def orthogonalize_head_read(attn_module, head_idx, r_unit):
    """Projects a direction out of head_idx's row slice of Wq, Wk, Wv."""
    D, H = attn_module.embed_dim, attn_module.num_heads
    hd = D // H
    r0, r1 = head_idx * hd, (head_idx + 1) * hd
    with torch.no_grad():
        W = attn_module.in_proj_weight.data
        for offset in (0, D, 2 * D):
            rows = W[offset + r0: offset + r1, :]
            coeff = rows @ r_unit
            W[offset + r0: offset + r1, :] = rows - torch.outer(coeff, r_unit)


def orthogonalize_mlp_write(mlp_module, P):
    """Projects a direction out of the MLP's last Linear layer."""
    lin_layers = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        lin_layers[-1].weight.data = P @ lin_layers[-1].weight.data
        if lin_layers[-1].bias is not None:
            lin_layers[-1].bias.data = P @ lin_layers[-1].bias.data


def orthogonalize_mlp_read(mlp_module, r_unit):
    """Projects a direction out of the MLP's first Linear layer."""
    lin_layers = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        W1 = lin_layers[0].weight.data
        coeff = W1 @ r_unit
        lin_layers[0].weight.data = W1 - torch.outer(coeff, r_unit)


def component_level_orthogonalize(model, r_hat, component_set, device, include_read=False, component_directions=None):
    """
    Removes a direction from only component_set's own weight matrices,
    leaving the rest of the model untouched.

    Args:
        model: not modified in place.
        r_hat: shared fallback direction, used unless component_directions overrides it.
        component_set: list of (layer, component) tuples.
        device: torch device.
        include_read: also orthogonalise read matrices, not just write.
        component_directions: optional dict (layer, component) -> direction,
            overrides r_hat for components present in it.

    Returns:
        a deep copy of model, orthogonalised.
    """
    model = deepcopy(model)
    r = r_hat.to(device)
    r_unit = r / (r.norm() + 1e-8)
    P = torch.eye(r_unit.size(0), device=device) - torch.outer(r_unit, r_unit)
    for (l, comp) in component_set:
        blk = model.transformer[l]
        if component_directions is not None and (l, comp) in component_directions:
            own_r = component_directions[(l, comp)].to(device)
            own_r_unit = own_r / (own_r.norm() + 1e-8)
            own_P = torch.eye(own_r_unit.size(0), device=device) - torch.outer(own_r_unit, own_r_unit)
        else:
            own_P, own_r_unit = P, r_unit
        if comp.startswith("head_"):
            h = int(comp.split("_")[1])
            orthogonalize_head_write(blk.attn, h, own_P)
            if include_read:
                orthogonalize_head_read(blk.attn, h, own_r_unit)
        else:
            orthogonalize_mlp_write(blk.mlp, own_P)
            if include_read:
                orthogonalize_mlp_read(blk.mlp, own_r_unit)
    return model


@torch.no_grad()
def evaluate_acc_asr_ra(model, loader, target_class, device, max_batches=None):
    """
    Evaluates ACC, ASR, and RA.

    Returns:
        dict with acc, asr, ra, n.
    """
    model.eval()
    correct_c = correct_b = correct_ra = total = 0
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
    if total == 0:
        return {"acc": 0.0, "asr": 0.0, "ra": 0.0, "n": 0}
    return {"acc": correct_c / total, "asr": correct_b / total, "ra": correct_ra / total, "n": total}


def phase2_results_table(results):
    """Builds a pandas DataFrame from compare_component_sets output, one row per condition."""
    import pandas as pd
    rows = []
    for name, r in results.items():
        rows.append({
            "condition": name,
            "ACC": round(r["acc"], 4),
            "ASR": round(r["asr"], 4),
            "RA": round(r["ra"], 4),
            "n": r["n"],
        })
    return pd.DataFrame(rows).set_index("condition")


def sweep_component_sets_by_threshold(model, loader, cumulative_history, component_directions, target_class, device,
                                       max_batches=None, save_dir=None):
    """
    Orthogonalisation analog of Phase 1's steering sweep, at the same
    thresholds and component sets Phase 1 already computed, reused
    directly from cumulative_history rather than recomputed. Steering is
    a temporary perturbation, orthogonalisation a permanent weight edit,
    so this checks whether the two actually saturate at the same
    threshold rather than assuming it.

    Cheap two-condition sweep (write-only, read-write), each component
    against its own direction. Once a good threshold is found, pull that
    one entry's cumulative_set and call compare_component_sets for the
    full breakdown. Caches unchanged consecutive sets, same as Phase 1.

    Args:
        model: not modified in place.
        loader: paired loader.
        cumulative_history: Phase 1's cumulative_component_steering history.
        component_directions: dict covering every component in cumulative_history,
            e.g. Phase 1's component_directions_cls.
        target_class: the attack's target class.
        device: torch device.
        max_batches: optional cap.
        save_dir: where to save the plot.

    Returns:
        dict with history and total_components.
    """
    save_dir = Path(save_dir) if save_dir is not None else Path("./results/graphs")
    save_dir.mkdir(parents=True, exist_ok=True)
    total_components = cumulative_history[-1]["set_size"] if cumulative_history else 0

    history = []
    prev_set_key, prev_result = None, None
    for entry in tqdm(cumulative_history, desc="Orthogonalisation sweep by cosine threshold"):
        t = entry["threshold"]
        component_set = entry["cumulative_set"]
        set_key = frozenset(component_set)

        if set_key == prev_set_key:
            write_metrics, read_metrics = prev_result
        else:
            if component_set:
                write_model = component_level_orthogonalize(model, component_directions[component_set[0]], component_set,
                                                              device, include_read=False, component_directions=component_directions)
                write_metrics = evaluate_acc_asr_ra(write_model, loader, target_class, device, max_batches)
                del write_model

                read_model = component_level_orthogonalize(model, component_directions[component_set[0]], component_set,
                                                             device, include_read=True, component_directions=component_directions)
                read_metrics = evaluate_acc_asr_ra(read_model, loader, target_class, device, max_batches)
                del read_model
            else:
                write_metrics = evaluate_acc_asr_ra(model, loader, target_class, device, max_batches)
                read_metrics = write_metrics
            prev_result = (write_metrics, read_metrics)
            prev_set_key = set_key

        history.append({
            "threshold": t, "set_size": len(component_set),
            "pct_components": 100.0 * len(component_set) / total_components if total_components else 0.0,
            "component_set": component_set,
            "write_acc": write_metrics["acc"], "write_asr": write_metrics["asr"], "write_ra": write_metrics["ra"],
            "read_acc": read_metrics["acc"], "read_asr": read_metrics["asr"], "read_ra": read_metrics["ra"],
        })

    _plot_sweep_component_sets_by_threshold(history, total_components, save_dir)
    return {"history": history, "total_components": total_components}


def _plot_sweep_component_sets_by_threshold(history, total_components, save_dir):
    """Three subplots: write-only, read-write, coverage. Same layout and x-axis as phase1's cumulative steering plot."""
    thresholds = [h["threshold"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))

    ax = axes[0]
    ax.plot(thresholds, [h["write_acc"] for h in history], "go-", lw=2, markersize=6, label="ACC")
    ax.plot(thresholds, [h["write_asr"] for h in history], "ro-", lw=2, markersize=6, label="ASR")
    ax.plot(thresholds, [h["write_ra"] for h in history], "bo-", lw=2, markersize=6, label="RA")
    ax.set_ylabel("Rate"); ax.set_ylim(-0.05, 1.08)
    ax.set_title("Write-only orthogonalisation, each component's own direction")

    ax = axes[1]
    ax.plot(thresholds, [h["read_acc"] for h in history], "go-", lw=2, markersize=6, label="ACC")
    ax.plot(thresholds, [h["read_asr"] for h in history], "ro-", lw=2, markersize=6, label="ASR")
    ax.plot(thresholds, [h["read_ra"] for h in history], "bo-", lw=2, markersize=6, label="RA")
    ax.set_ylabel("Rate"); ax.set_ylim(-0.05, 1.08)
    ax.set_title("Read and write orthogonalisation, each component's own direction")

    for ax in axes[:2]:
        ax.set_xlabel("Cosine similarity threshold (|cosine_sim| >= threshold)")
        ax.invert_xaxis()
        ax.set_xticks([round(t, 1) for t in np.arange(1.0, -0.01, -0.1)])
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.step(thresholds, [h["set_size"] for h in history], where="post", color="steelblue", lw=2)
    ax.set_xlabel("Cosine similarity threshold (|cosine_sim| >= threshold)")
    ax.set_ylabel("Number of components included")
    ax.set_ylim(-0.5, total_components + 0.5)
    ax.invert_xaxis()
    ax.set_xticks([round(t, 1) for t in np.arange(1.0, -0.01, -0.1)])
    ax.set_title(f"Coverage: number of components included, out of {total_components} total")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Weight orthogonalisation by cosine similarity threshold",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "phase2_threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()


def compare_component_sets(model, loader, r_hat, component_sets, target_class, device, max_batches=None,
                            include_read_variant=True, component_directions=None):
    """
    Evaluates several named component sets against a shared baseline and
    whole-model reference.

    Scope (narrow vs whole model) and direction source (shared r_hat vs
    own direction) are independent axes:
        narrow, shared r_hat:  {name}_write_only, {name}_read_write
        narrow, own direction: {name}_own_direction_write_only, {name}_own_direction_read_write
        whole model, shared:   whole_model
        whole model, own:      {name}_whole_model_top_direction, {name}_whole_model_joint_direction

    top_direction uses the set's own highest-ranked component alone,
    joint_direction removes the shared subspace of every component in
    that set. Both computed per named set, not globally shared.

    Args:
        model: not modified in place.
        loader: paired loader.
        r_hat: shared direction from phase0.
        component_sets: dict name -> list of (layer, component) tuples.
        target_class: the attack's target class.
        device: torch device.
        max_batches: optional cap.
        include_read_variant: also run each read-write condition.
        component_directions: optional dict (layer, component) -> direction,
            shared across every named set.

    Returns:
        dict of condition -> evaluate_acc_asr_ra output. Pass to phase2_results_table.
    """
    results = {"baseline": evaluate_acc_asr_ra(model, loader, target_class, device, max_batches)}

    whole_model_model = global_orthogonalize(model, r_hat, device)
    results["whole_model"] = evaluate_acc_asr_ra(whole_model_model, loader, target_class, device, max_batches)
    del whole_model_model

    for name, component_set in component_sets.items():
        write_model = component_level_orthogonalize(model, r_hat, component_set, device, include_read=False)
        results[f"{name}_write_only"] = evaluate_acc_asr_ra(write_model, loader, target_class, device, max_batches)
        del write_model

        if include_read_variant:
            read_model = component_level_orthogonalize(model, r_hat, component_set, device, include_read=True)
            results[f"{name}_read_write"] = evaluate_acc_asr_ra(read_model, loader, target_class, device, max_batches)
            del read_model

        own_directions = [component_directions[k] for k in component_set if component_directions and k in component_directions]

        if own_directions:
            own_dir_write_model = component_level_orthogonalize(model, r_hat, component_set, device, include_read=False,
                                                                  component_directions=component_directions)
            results[f"{name}_own_direction_write_only"] = evaluate_acc_asr_ra(own_dir_write_model, loader, target_class, device, max_batches)
            del own_dir_write_model

            own_dir_read_model = component_level_orthogonalize(model, r_hat, component_set, device, include_read=True,
                                                                 component_directions=component_directions)
            results[f"{name}_own_direction_read_write"] = evaluate_acc_asr_ra(own_dir_read_model, loader, target_class, device, max_batches)
            del own_dir_read_model

            whole_model_top = global_orthogonalize(model, own_directions[0], device)
            results[f"{name}_whole_model_top_direction"] = evaluate_acc_asr_ra(whole_model_top, loader, target_class, device, max_batches)
            del whole_model_top

            if len(own_directions) > 1:
                whole_model_joint = global_orthogonalize_multi(model, own_directions, device)
                results[f"{name}_whole_model_joint_direction"] = evaluate_acc_asr_ra(whole_model_joint, loader, target_class, device, max_batches)
                del whole_model_joint

    return results