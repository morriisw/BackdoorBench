"""
Phase 2: Component-Level Weight Orthogonalisation (proposal Section 4.1.6, Task 3).

Extends phase0's global_orthogonalize (whole-model, layer-level) down to the
individual components identified in Phase 1, and compares the two directly
on ACC, ASR, and RA. This is the real weight-space defence, not the
activation-hook preview from phase1's compare_layer_vs_component_defense_proxy.

Two component-level variants are provided:
  write_only: orthogonalises only the matrices that WRITE a component's own
    contribution into the residual stream (a head's own columns of
    out_proj, and the MLP's last Linear). This is mathematically identical
    to phase1's activation-level component proxy (projecting a weight
    matrix's output is the same operation as projecting that output
    directly), so it should reproduce those numbers exactly and is a good
    correctness check on this implementation.
  read_write: additionally orthogonalises the matrices that READ into a
    component (a head's own rows of Wq, Wk, Wv, and the MLP's first
    Linear). This changes what the component computes, not just what it
    writes out, something no activation-level projection can represent, so
    it is the only variant that can produce results different from the
    activation proxy already measured in phase1.
"""
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np


def orthogonalize_head_write(attn_module, head_idx, P):
    """
    Removes r_hat from head_idx's own contribution to the residual stream
    by projecting its column slice of out_proj.weight: W_new = P @ W,
    restricted to the columns belonging to this head. Bias is left
    untouched here, since out_proj.bias is added once after all heads are
    summed and does not belong to any single head.
    """
    D, H = attn_module.embed_dim, attn_module.num_heads
    hd = D // H
    c0, c1 = head_idx * hd, (head_idx + 1) * hd
    with torch.no_grad():
        Wo = attn_module.out_proj.weight.data
        Wo[:, c0:c1] = P @ Wo[:, c0:c1]


def orthogonalize_head_read(attn_module, head_idx, r_unit):
    """
    Removes r_hat from what head_idx's Q, K, and V are allowed to depend
    on, by right-multiplying its row slice of the combined in_proj_weight
    (stacked Wq, Wk, Wv): W_new = W(I - r_hat r_hat^T). Unlike the write
    side, this decomposes cleanly per head, since each row's correction
    only depends on that row's own dot product with r_hat.
    """
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
    """Removes r_hat from the MLP's own output, by projecting the last Linear layer's weight and bias."""
    lin_layers = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        lin_layers[-1].weight.data = P @ lin_layers[-1].weight.data
        if lin_layers[-1].bias is not None:
            lin_layers[-1].bias.data = P @ lin_layers[-1].bias.data


def orthogonalize_mlp_read(mlp_module, r_unit):
    """Removes r_hat from what the MLP's hidden layer can depend on, by right-multiplying the first Linear layer's weight."""
    lin_layers = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        W1 = lin_layers[0].weight.data
        coeff = W1 @ r_unit
        lin_layers[0].weight.data = W1 - torch.outer(coeff, r_unit)


def component_level_orthogonalize(model, r_hat, component_set, device, include_read=False):
    """
    Returns a deep copy of model with r_hat removed from only the weight
    matrices belonging to component_set. include_read=False (default)
    touches write matrices only, and will numerically match phase1's
    activation-level component proxy. include_read=True also touches each
    component's read matrices, the only way this can diverge from that
    proxy.
    """
    model = deepcopy(model)
    r = r_hat.to(device)
    r_unit = r / (r.norm() + 1e-8)
    P = torch.eye(r_unit.size(0), device=device) - torch.outer(r_unit, r_unit)
    for (l, comp) in component_set:
        blk = model.transformer[l]
        if comp.startswith("head_"):
            h = int(comp.split("_")[1])
            orthogonalize_head_write(blk.attn, h, P)
            if include_read:
                orthogonalize_head_read(blk.attn, h, r_unit)
        else:
            orthogonalize_mlp_write(blk.mlp, P)
            if include_read:
                orthogonalize_mlp_read(blk.mlp, r_unit)
    return model


def layer_wide_orthogonalize(model, r_hat, device):
    """
    Removes r_hat from every writing matrix in the model: the patch-
    embedding Linear, and every layer's out_proj and MLP-last-Linear. This
    reproduces phase0's global_orthogonalize exactly, the real layer-level
    baseline defence this file compares against.
    """
    model = deepcopy(model)
    r = r_hat.to(device)
    r_unit = r / (r.norm() + 1e-8)
    P = torch.eye(r_unit.size(0), device=device) - torch.outer(r_unit, r_unit)
    with torch.no_grad():
        for m in model.to_patch_embedding.modules():
            if isinstance(m, nn.Linear) and m.weight.size(0) == r_unit.size(0):
                m.weight.data = P @ m.weight.data
                if m.bias is not None:
                    m.bias.data = P @ m.bias.data
        for blk in model.transformer:
            blk.attn.out_proj.weight.data = P @ blk.attn.out_proj.weight.data
            if blk.attn.out_proj.bias is not None:
                blk.attn.out_proj.bias.data = P @ blk.attn.out_proj.bias.data
            lin_layers = [m for m in blk.mlp.modules() if isinstance(m, nn.Linear)]
            lin_layers[-1].weight.data = P @ lin_layers[-1].weight.data
            if lin_layers[-1].bias is not None:
                lin_layers[-1].bias.data = P @ lin_layers[-1].bias.data
    return model


@torch.no_grad()
def evaluate_acc_asr_ra(model, loader, target_class, device, max_batches=None):
    """Evaluates (ACC, ASR, RA) with binomial standard error, matching phase0/phase1's metric definitions."""
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
        return {"acc": 0.0, "asr": 0.0, "ra": 0.0, "acc_se": 0.0, "asr_se": 0.0, "ra_se": 0.0, "n": 0}

    def se(p, n):
        return float(np.sqrt(p * (1 - p) / n))

    acc, asr, ra = correct_c / total, correct_b / total, correct_ra / total
    return {"acc": acc, "asr": asr, "ra": ra, "acc_se": se(acc, total), "asr_se": se(asr, total), "ra_se": se(ra, total), "n": total}


def phase2_results_table(results):
    """
    Builds a pandas DataFrame from run_phase2_comparison's output, one row
    per condition, columns for ACC, ASR, RA each with its standard error in
    a separate column. Renders as a clean table in a notebook cell.
    """
    import pandas as pd
    rows = []
    for name, r in results.items():
        rows.append({
            "condition": name,
            "ACC": round(r["acc"], 4), "ACC_se": round(r["acc_se"], 4),
            "ASR": round(r["asr"], 4), "ASR_se": round(r["asr_se"], 4),
            "RA": round(r["ra"], 4), "RA_se": round(r["ra_se"], 4),
            "n": r["n"],
        })
    return pd.DataFrame(rows).set_index("condition")


def run_phase2_comparison(model, loader, r_hat, component_set, target_class, device, max_batches=None, include_read_variant=True):
    """
    Runs the Task 3 comparison: no defence, whole-model weight
    orthogonalisation (the Karayalcin-style baseline), component-level
    write-only orthogonalisation, and optionally component-level
    read-and-write orthogonalisation. Returns a dict of (ACC, ASR, RA) with
    standard error for each condition. Call phase2_results_table on the
    return value for a clean table view in a notebook.
    """
    results = {"baseline": evaluate_acc_asr_ra(model, loader, target_class, device, max_batches)}
 
    layer_model = layer_wide_orthogonalize(model, r_hat, device)
    results["layer_wide"] = evaluate_acc_asr_ra(layer_model, loader, target_class, device, max_batches)
    del layer_model
 
    write_model = component_level_orthogonalize(model, r_hat, component_set, device, include_read=False)
    results["component_write_only"] = evaluate_acc_asr_ra(write_model, loader, target_class, device, max_batches)
    del write_model
 
    if include_read_variant:
        read_model = component_level_orthogonalize(model, r_hat, component_set, device, include_read=True)
        results["component_read_write"] = evaluate_acc_asr_ra(read_model, loader, target_class, device, max_batches)
        del read_model
 
    return results