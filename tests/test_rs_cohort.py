import json

from examples.mpc.franka.experiments.ik2.test_rs import csv_source_candidates
from examples.mpc.franka.experiments.ik2.run_rs_cohort import export_cohort, update_paper


def test_csv_candidates_filter_task_success_and_duplicate_seed(tmp_path):
    recent = tmp_path/'recent.csv'
    recent.write_text('task,object,seed,success\nrotation,mug,0,1\nrotation,mug,1,0\n'
                      'flipping,mug,2,1\nrotation,mug,0,1\nrotation,teapot,3,1\n')
    old = tmp_path/'old.csv'
    old.write_text('object,seed,success\nmug,2,1\nmug,4,True\n')
    rotation = csv_source_candidates([recent, old], obj='mug', task='rotation', seed=0)
    assert [c['trial'] for c in rotation] == [0]
    flipping = csv_source_candidates([recent, old], obj='mug', task='flipping', seed=0)
    assert sorted(c['trial'] for c in flipping) == [2, 4]
    failed = csv_source_candidates([recent, old], obj='mug', task='rotation', seed=0, success=False)
    assert [c['trial'] for c in failed] == [1]


def test_cohort_exports_complete_cases_only_and_updates_target_table(tmp_path):
    source = dict(success=True, steps=100, policy_interval_s=.02,
                  rs_source_selected=True, rs_binary_decisions=100,
                  rs_near_selections=75, rs_optimal_selections=25,
                  rs_evaluations=100, rs_switch_events=10, mpc_contact_switches=2,
                  p_star_patch_switches=4)
    states = [dict(advantage_rs=a, cdr=max(0, -a), event_kind='decision') for a in [0, .2]]
    complete = tmp_path/'complete'
    complete.mkdir()
    (complete/'trials.jsonl').write_text(json.dumps(source)+'\n')
    (complete/'states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in states))
    cases = [dict(task='rotation', status='complete', output=str(complete)),
             dict(task='rotation', status='failed', output=str(tmp_path/'failed'))]
    result, rows = export_cohort(tmp_path, ['rotation'], cases, .99)
    m = result['by_task']['rotation']['table_vi']
    assert m['sampled_points'] == 2 and m['p_advantage_positive'] == .5
    assert m['f_near_hz'] == 37.5 and m['f_p_star_hz'] == 2
    assert rows[0].count(' & ') == 7
    paper = tmp_path/'paper.tex'
    paper.write_text('Rotation & unrelated\\\n\\label{tab:ranking_advantage}\n'
                     'Rotation & -- \\\\\nFlipping & -- \\\\\n\\end{table}\n')
    update_paper(paper, rows)
    assert paper.read_text().startswith('Rotation & unrelated')
    assert rows[0] in paper.read_text()
    assert 'Flipping & --' in paper.read_text()
