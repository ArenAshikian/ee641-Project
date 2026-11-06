"""Archive the pre-registered October Maia result without touching any model."""

import json
import shutil
from datetime import datetime, timezone

from chess_rl.ppo_train import ROOT, CHECKPOINTS, OUTPUTS, sha256, V0

FILES = [
    OUTPUTS / 'maia_confirm_oct2026_manifest.json',
    OUTPUTS / 'maia_confirm_oct2026_games.jsonl',
    OUTPUTS / 'maia_confirm_oct2026_summary.json',
    ROOT / 'data' / 'processed' / 'maia_confirmation_oct2026.json',
    ROOT / 'chess_rl' / 'maia_confirm.py',
    ROOT / 'chess_rl' / 'maia_benchmark.py',
    ROOT / 'README_MAIA_CONFIRM.md',
]
MODELS = {
    'V0 supervised': V0,
    'V100 initial PPO': CHECKPOINTS / 'ppo_v0100.pt',
    'Latest S49 continued PPO': CHECKPOINTS / 'ppo_scale_latest' / 's0049.pt',
}


def archive():
    missing = [str(f) for f in FILES if not f.is_file()]
    missing += [str(p) for p in MODELS.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Cannot archive incomplete results: ' + '\n'.join(missing))
    summary = json.loads((OUTPUTS / 'maia_confirm_oct2026_summary.json').read_text(encoding='utf-8'))
    test = summary.get('primary_test')
    if not summary.get('completed') or not test or test.get('matched_pairs') != 256:
        raise RuntimeError('October confirmation report is not complete or primary test missing.')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    folder = ROOT / 'research_archive' / f'october_maia_confirmation_{stamp}'
    folder.mkdir(parents=True, exist_ok=False)
    records = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'description': 'Preserved October Maia pre-registered confirmation result',
        'copied_files': {}, 'checkpoint_references': {}, 'primary_test': test,
        'warning': ('Model files were hashed, not copied. Separately back up '
                    'the checkpoints folder for a full recoverable archive.'),
    }
    for src in FILES:
        rel = src.relative_to(ROOT)
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records['copied_files'][str(rel)] = {'sha256': sha256(dst), 'bytes': dst.stat().st_size}
    for label, model in MODELS.items():
        records['checkpoint_references'][label] = {'file': str(model), 'sha256': sha256(model)}
    (folder / 'archive_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Research evidence archived at:', folder)
    print('Primary test:', test)
    print('Original results and model checkpoints were not modified.')
    return folder


if __name__ == '__main__':
    archive()
