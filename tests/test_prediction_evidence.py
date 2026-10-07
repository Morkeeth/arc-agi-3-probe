import importlib.util
import sys
from pathlib import Path
import pytest

P = Path(__file__).resolve().parents[1] / 'kaggle/ab-compact/prediction_evidence.py'
spec = importlib.util.spec_from_file_location('prediction_evidence', P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def args():
    return dict(game_id='g', level=2, call_id='call-1', rule_id='row-major',
                before={'grid': [[1, 2], [3, 4]], 'phase': 'filled'}, action={'name': 'SPACE'},
                prediction={'level_completed': True})


def observation(a, outcome):
    return {k: v for k, v in a.items() if k not in ('rule_id', 'prediction')} | {'outcome': outcome}


def test_mismatch_preserves_exact_state_action_outcome_without_global_rejection():
    book = m.PredictionEvidence()
    a = args()
    book.commit(**a)
    result = book.observe(**observation(a, {'level_completed': False, 'board_changed': True}))
    assert result['verdict'] == 'MISMATCH'
    assert result['differences'] == {'level_completed': {'predicted': True, 'observed': False}}
    assert result['commit_sequence'] < result['observation_sequence']
    assert book.retrieve(game_id='g', level=2, before=a['before'])
    assert not book.retrieve(game_id='other', level=2, before=a['before'])
    assert not book.retrieve(game_id='g', level=3, before=a['before'])
    assert not book.retrieve(game_id='g', level=2, before={'grid': [[1]], 'phase': 'filled'})
    assert not book.retrieve(game_id='g', level=2, before=a['before'], max_chars=1)


def test_missing_observation_is_not_false_prediction():
    book = m.PredictionEvidence(); a = args(); book.commit(**a)
    result = book.observe(**observation(a, {'board_changed': True}))
    assert result['verdict'] == 'INCOMPLETE'
    assert not book.retrieve(game_id='g', level=2, before=a['before'])


def test_cannot_predict_after_seeing_outcome_or_rewrite_commitment():
    book = m.PredictionEvidence(); a = args()
    with pytest.raises(ValueError, match='prior commitment'):
        book.observe(**observation(a, {'level_completed': False}))
    with pytest.raises(ValueError, match='already'):
        book.commit(**a)
    a['call_id'] = 'new'; book.commit(**a)
    with pytest.raises(ValueError, match='already'):
        book.commit(**a)


@pytest.mark.parametrize('changed', ['before', 'action', 'level'])
def test_wrong_board_action_or_level_cannot_validate_prediction(changed):
    book = m.PredictionEvidence(); a = args(); book.commit(**a)
    obs = observation(a, {'level_completed': True})
    obs[changed] = 9 if changed == 'level' else {'wrong': True}
    with pytest.raises(ValueError, match='differs'):
        book.observe(**obs)
    assert not book.records


def test_no_vacuous_prediction_or_boolean_number_equivalence():
    book = m.PredictionEvidence(); a = args()
    with pytest.raises(ValueError, match='vacuously'):
        book.commit(**(a | {'prediction': {}}))
    book.commit(**a)
    result = book.observe(**observation(a, {'level_completed': 1}))
    assert result['verdict'] == 'MISMATCH'


def test_mutating_inputs_after_commit_cannot_change_prior_prediction():
    book = m.PredictionEvidence(); a = args(); book.commit(**a)
    a['prediction']['level_completed'] = False
    result = book.observe(**observation(a, {'level_completed': False}))
    assert result['verdict'] == 'MISMATCH'
    result['outcome']['level_completed'] = True
    assert book.records[0]['outcome']['level_completed'] is False
