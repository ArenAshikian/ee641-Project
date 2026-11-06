"""Checkpoint format reserved for supervised and PPO training.

After V0, archive V10, V20, V30, and so on permanently.
"""

from pathlib import Path

import torch


def is_archive_version(version):
    return version == 0 or version % 10 == 0


def save_checkpoint(model, version, optimizer=None, folder="checkpoints", metadata=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "version": version,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metadata": metadata or {},
    }
    # Never overwrite a permanent milestone, even if a run is restarted.
    if is_archive_version(version):
        archive = folder / f"v{version:04d}.pt"
        if archive.exists():
            raise FileExistsError(f"Permanent checkpoint already exists: {archive}")
        torch.save(state, archive)
    # Latest is deliberately replaced after every training iteration.
    torch.save(state, folder / "latest.pt")
    return folder / "latest.pt"
