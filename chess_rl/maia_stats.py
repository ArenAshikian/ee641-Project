"""Small, dependency-light paired-opening tournament statistics."""
import numpy as np


def score_from_record(record):
    if record.get('truncated'):
        return None
    return record.get('score')


def summarize_games(records, expected_pairs, seed=2026, bootstrap=2000):
    """Keep truncated games out of scores AND their full paired opening out of CI."""
    wins = sum(r['score'] == 1.0 for r in records if not r['truncated'])
    draws = sum(r['score'] == 0.5 for r in records if not r['truncated'])
    losses = sum(r['score'] == 0.0 for r in records if not r['truncated'])
    truncated = sum(bool(r['truncated']) for r in records)
    played = wins + draws + losses
    pairs = {}
    for row in records:
        pairs.setdefault(int(row['opening_id']), {})[row['color']] = score_from_record(row)
    values = []
    for pair in pairs.values():
        if ('white' in pair and 'black' in pair and
                pair['white'] is not None and pair['black'] is not None):
            values.append((float(pair['white']) + float(pair['black'])) / 2)
    ci = paired_bootstrap(values, seed=seed, bootstrap=bootstrap)
    return {
        'wins': wins, 'draws': draws, 'losses': losses, 'truncated': truncated,
        'total_games': len(records), 'completed_games': played,
        'expected_games': expected_pairs*2,
        'score_completed': round((wins + draws*.5)/played, 5) if played else None,
        'complete_pairs': len(values), 'ci95_paired': ci,
    }


def paired_bootstrap(values, seed=2026, bootstrap=2000):
    if len(values) < 5:
        return None
    numbers = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # Avoid large intermediate allocations for larger evaluations.
    means = []
    for _ in range(bootstrap):
        means.append(float(numbers[rng.integers(0, len(numbers), size=len(numbers))].mean()))
    return [round(float(x), 5) for x in np.quantile(means, [.025, .975])]


def compare_models(a_records, b_records, seed=2026):
    """Paired bootstrap differences across matching opening IDs and BOTH colors."""
    def complete_pairs(rows):
        by_pair = {}
        for row in rows:
            by_pair.setdefault(int(row['opening_id']), {})[row['color']] = score_from_record(row)
        return {i: (sum(scores.values())/2) for i, scores in by_pair.items()
                if set(scores) == {'white', 'black'} and all(v is not None for v in scores.values())}
    a, b = complete_pairs(a_records), complete_pairs(b_records)
    indices = sorted(set(a) & set(b))
    if not indices:
        return {'matched_opening_pairs': 0, 'score_delta': None, 'ci95_paired_difference': None}
    differences = [a[i] - b[i] for i in indices]
    return {'matched_opening_pairs': len(indices),
            'score_delta': round(float(np.mean(differences)), 5),
            'ci95_paired_difference': paired_bootstrap(differences, seed=seed)}
