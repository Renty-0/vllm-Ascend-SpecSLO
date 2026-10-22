"""Uninstalled, target-blind B4 selector prototype for grouped HOST exploration.

Training labels come from previous completed observations. A current request's
target heads and acceptance outcome are never arguments to predict/choose.
This module has no model, runtime installation, or gain-table entry point.
"""
import itertools
import math

PARENTS = (-1, 0, 1, 2, -1, 0, 1, 2)
SPINE = (0, 1, 2, 3)
LAYOUTS = tuple(
    nodes for nodes in itertools.combinations(range(8), 4)
    if all(PARENTS[i] < 0 or PARENTS[i] in nodes for i in nodes)
)
PRIOR_STRENGTH = 4.0
OTHER_CLASS_WEIGHT = 0.25


def probability_bin(primary):
    return int(primary >= 0.6) + int(primary >= 0.9)


def make_observations(completed_rows):
    observations = []
    for row in completed_rows:
        mismatch = row['first_old_primary_mismatch']
        for depth in range(4 if mismatch is None else mismatch + 1):
            outcome = 0 if mismatch is None or depth < mismatch else (
                1 if row['available_top2_side_match'] else 2
            )
            probabilities = row['raw_conditional_probabilities']
            observations.append(dict(
                request_id=row['request_id'], category=row['category'], depth=depth,
                probability_bin=probability_bin(probabilities[depth]), outcome=outcome,
            ))
    return tuple(observations)


def predict_rank_probabilities(category, probabilities, previous_observations):
    if len(probabilities) != 8:
        raise ValueError('eight current draft probabilities required')
    result = []
    for depth in range(4):
        primary, side = probabilities[depth], probabilities[depth + 4]
        if not (math.isfinite(primary) and math.isfinite(side)
                and 0 <= side <= primary <= 1 and primary + side <= 1.00001):
            raise ValueError('finite actual top2 probabilities required')
        counts = [PRIOR_STRENGTH * primary, PRIOR_STRENGTH * side,
                  PRIOR_STRENGTH * max(0.0, 1 - primary - side)]
        for observation in previous_observations:
            if observation['depth'] == depth and observation['probability_bin'] == probability_bin(primary):
                weight = 1.0 if observation['category'] == category else OTHER_CLASS_WEIGHT
                counts[observation['outcome']] += weight
        total = sum(counts)
        result.append(tuple(value / total for value in counts))
    return tuple(result)


def choose(category, current_draft_probabilities, previous_observations):
    rank_probabilities = predict_rank_probabilities(
        category, current_draft_probabilities, previous_observations
    )
    node_probabilities = [p[0] for p in rank_probabilities] + [p[1] for p in rank_probabilities]
    path_probabilities = []
    for parent, probability in zip(PARENTS, node_probabilities):
        path_probabilities.append(probability * (path_probabilities[parent] if parent >= 0 else 1.0))
    # A leaf's one correction/bonus token is common to each nonempty layout.
    # Scores predict acceptance, not latency or workload Goodput.
    scores = {nodes: sum(path_probabilities[i] for i in nodes) for nodes in LAYOUTS}
    nodes = max(LAYOUTS, key=lambda layout: (scores[layout], layout == SPINE, tuple(-i for i in layout)))
    return nodes, dict(rank_probabilities=rank_probabilities,
                      expected_accepted_sum=scores[nodes],
                      expected_spine_accepted_sum=scores[SPINE],
                      predicted_delta_vs_spine=scores[nodes] - scores[SPINE])
