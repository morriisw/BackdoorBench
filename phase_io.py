"""
Tiny save/load helpers so any phase's results dict can be cached to
disk and reloaded after a kernel restart, without re-running the
pipeline call that produced it. Saves alongside that phase's own plots,
in the same save_dir already used by get_graph_dir, so everything for
one model/attack/dataset combination stays in one folder.
"""
import torch
from pathlib import Path


def save_phase_results(results, save_dir, filename, exclude_keys=None):
    """
    Saves a results dict to save_dir/filename.

    Args:
        results: the dict returned by a run_phaseN_pipeline call.
        save_dir: the same directory that phase's plots were saved to.
        filename: for example "phase0_results.pt".
        exclude_keys: iterable of keys to drop before saving. Useful for
            anything not worth persisting.

    Returns:
        the path saved to.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    exclude_keys = set(exclude_keys or [])
    to_save = {k: v for k, v in results.items() if k not in exclude_keys}
    path = save_dir / filename
    torch.save(to_save, path)
    print(f"Saved -> {path}")
    return path


def load_phase_results(save_dir, filename, map_location="cpu"):
    """
    Loads a results dict previously written by save_phase_results, or
    returns None if it does not exist yet.

    Args:
        save_dir: same directory passed to save_phase_results.
        filename: same filename passed to save_phase_results.
        map_location: torch.load's map_location, "cpu" by default.

    Returns:
        the results dict, or None if the file does not exist.
    """
    path = Path(save_dir) / filename
    if not path.exists():
        return None
    results = torch.load(path, map_location=map_location, weights_only=False)
    print(f"Loaded <- {path}")
    return results