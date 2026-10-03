"""Exact pilot retrieval, optional local matching, and labelled-only metrics."""
from dataclasses import asdict, dataclass
import time

import numpy as np

from app.services.coat_color import mismatch
from app.services.lost_search import auxiliary_evidence
from .dataset import optional_date


@dataclass(frozen=True)
class Ranking:
    name: str
    animal_weight: float = .7
    color_weight: float = .1
    patch_weight: float = 0.
    pool: int = 200
    date_policy: str = 'current'

    def __post_init__(self):
        if not 0 <= self.animal_weight <= 1 or not 0 <= self.color_weight <= .2 or not 0 <= self.patch_weight <= 1:
            raise ValueError('Invalid ranking weights')
        if type(self.pool) is not int or not 1 <= self.pool <= 300 or self.date_policy not in {'current', 'filter'}:
            raise ValueError('Invalid candidate/date policy')


RANKINGS = (Ranking('current'), Ranking('animal-only', animal_weight=1.),
            Ranking('patch-rerank', patch_weight=.2), Ranking('date-policy', date_policy='filter'))


def eligible(query, candidate, settings):
    if query.species != candidate.species or query.partition != candidate.partition:
        return False
    statuses = {None, 'NOTICE', 'PROTECT'}
    if query.metadata.get('include_adopted_or_returned'):
        statuses.update(('ADOPTED', 'RETURNED'))
    if candidate.metadata.get('status') not in statuses:
        return False
    lost = optional_date(query.metadata.get('lost_date'))
    found = optional_date(candidate.metadata.get('happen_date'))
    return not (settings.date_policy == 'filter' and lost and found and found < lost)


def patch_match(query, candidate):
    """Mutual nearest neighbours with bidirectional coverage; no body-part labels.

    Thresholds are experimental guards, not calibrated same-animal probabilities.
    A single fur patch cannot establish agreement over an animal.
    """
    q, g = query['patches'], candidate['patches']
    empty = {'score': None, 'matches': 0, 'query_coverage': 0., 'candidate_coverage': 0., 'pairs': []}
    if len(q) < 4 or len(g) < 4:
        return empty
    similarities = np.clip(q @ g.T, -1., 1.)
    forward, backward = similarities.argmax(axis=1), similarities.argmax(axis=0)
    pairs = [(i, int(j)) for i, j in enumerate(forward) if backward[j] == i]
    qcoverage, gcoverage = len(pairs) / len(q), len(pairs) / len(g)
    enough = len(pairs) >= 4 and min(qcoverage, gcoverage) >= .25
    score = float(np.mean([similarities[i, j] for i, j in pairs])) if enough else None
    strongest = sorted(pairs, key=lambda pair: -float(similarities[pair]))[:12]
    return {'score': score, 'matches': len(pairs), 'query_coverage': qcoverage, 'candidate_coverage': gcoverage,
            'pairs': [{'query_xy': query['coordinates'][i].tolist(), 'candidate_xy': candidate['coordinates'][j].tolist(),
                       'cosine': float(similarities[i, j])} for i, j in strongest]}


def evaluate_query(query, gallery, get_features, settings):
    started = time.perf_counter()
    q, qm = get_features(query)
    rows = []
    by_id = {p.id: p for p in gallery}
    for candidate in gallery:
        if not eligible(query, candidate, settings):
            continue
        c, cm = get_features(candidate)
        full = float(np.clip(q['full'] @ c['full'], -1., 1.))
        animal = float(np.clip(q['animal'] @ c['animal'], -1., 1.)) if len(q['animal']) and len(c['animal']) else None
        image = full if animal is None else (1 - settings.animal_weight) * full + settings.animal_weight * animal
        rows.append({'image_id': candidate.id, 'animal_id': candidate.metadata['animal_id'],
                     'full_score': full, 'animal_score': animal, 'image_score': image,
                     'focus_status': cm['focus_status'], 'animal_fallback': animal is None})
    rows.sort(key=lambda row: (-row['image_score'], row['animal_id']))
    positive_ids = {p.id for p in gallery if query.identity and p.identity == query.identity}
    first_rank = next((i for i, row in enumerate(rows, 1) if row['image_id'] in positive_ids), None)
    selected = rows[:settings.pool]
    for row in selected:
        candidate = by_id[row['image_id']]
        c, cm = get_features(candidate)
        evidence = auxiliary_evidence(candidate.metadata, optional_date(query.metadata.get('lost_date')),
                                      query.metadata.get('region'), query.metadata.get('query_description'))
        distance = mismatch(qm.get('coat_color'), cm.get('coat_color'))
        penalty = 0. if distance is None else settings.color_weight * distance
        local = patch_match(q, c) if settings.patch_weight else None
        score = row['image_score']
        if local and local['score'] is not None:
            score = (1 - settings.patch_weight) * score + settings.patch_weight * local['score']
        row.update(metadata_boost=.01 * len(evidence), color_penalty=penalty,
                   color_available=distance is not None, matched_evidence=evidence, patch=local,
                   final_score=score + .01 * len(evidence) - penalty)
    selected.sort(key=lambda row: (-row['final_score'], -row['image_score'], row['animal_id']))
    final_rank = next((i for i, row in enumerate(selected, 1) if row['image_id'] in positive_ids), None)
    return {'query_id': query.id, 'truth': query.truth, 'identity': query.identity, 'partition': query.partition,
            'ranking': asdict(settings), 'query_focus_status': qm['focus_status'],
            'eligible_count': len(rows), 'candidate_count': len(selected),
            'pool_truncated': len(rows) > settings.pool,
            'first_positive_rank': first_rank if query.truth == 'present' else None,
            'final_positive_rank': final_rank if query.truth == 'present' else None,
            'top20': selected[:20], 'evaluation_seconds_including_local_feature_reads': time.perf_counter() - started}


def metrics(results):
    output = {}
    for partition in ('tune', 'holdout', 'case'):
        cohort = [r for r in results if r['partition'] == partition]
        present = [r for r in cohort if r['truth'] == 'present']
        absent = [r for r in cohort if r['truth'] == 'absent']
        count = len(present)
        output[partition] = {'queries': len(cohort), 'verified_present_queries': count,
            'verified_identities': len({r['identity'] for r in present}), 'verified_absent_queries': len(absent),
            'unverified_queries': sum(r['truth'] == 'unknown' for r in cohort),
            'first_stage_recall': sum(r['first_positive_rank'] is not None and r['first_positive_rank'] <= r['ranking']['pool'] for r in present) / count if count else None,
            'recall_at': {str(k): sum(r['final_positive_rank'] is not None and r['final_positive_rank'] <= k for r in present) / count if count else None for k in (1, 5, 10, 20)},
            'absent_max_scores': [r['top20'][0]['final_score'] if r['top20'] else None for r in absent],
            'truncated_queries': sum(r['pool_truncated'] for r in cohort)}
    return output
