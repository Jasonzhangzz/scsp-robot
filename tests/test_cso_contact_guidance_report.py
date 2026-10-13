import pytest

from examples.mpc.fingertips.test.run_cso_contact_guidance import (
    MODELS,OBJECTS,fill_table,validate_cohort,
)


def report():
    return dict(trials=[dict(model=m,object=o,seed=s) for m in MODELS for o in OBJECTS for s in range(5)],
                models={m:dict(total_trials=35,success_rate=34/35,
                    execution_time_s_success_only=dict(n=34,mean=2.,std=.5),
                    planning_time_ms_all_rounds=dict(n=100,mean=3.,std=.25)) for m in MODELS})


def test_incomplete_cohort_cannot_fill_the_paper():
    r=report();r['trials'].pop()
    with pytest.raises(ValueError,match='Incomplete'):
        validate_cohort(r)
    r=report();r['trials'][-1]=r['trials'][0]
    with pytest.raises(ValueError,match='duplicate'):
        validate_cohort(r)


def test_fill_model_cells_preserves_solver_cells_and_other_tables():
    text=r'''Unrelated text
\begin{table}[t]
\caption{Old caption}
\label{tab:cso_contact_guidance}
Success Rate & 100\% & 100\% & 100\% & keep1 & keep2 & keep3 & keep4 \\
Execution Time (s) & -- & -- & -- & 1 & 2 & 3 & 4 \\
Planning Time (ms) & -- & -- & -- & a & b & c & d \\
\end{table}
Other table 100\%
'''
    out=fill_table(text,report())
    assert out.count(r'97.1\%')==3
    assert out.count(r'$2.00 \pm 0.50$')==3
    assert out.count(r'$3.00 \pm 0.25$')==3
    assert 'keep1 & keep2 & keep3 & keep4' in out
    assert 'Unrelated text' in out and r'Other table 100\%' in out


def test_no_successes_do_not_become_zero_second_completions():
    r=report();r['models']['lcp']['execution_time_s_success_only']=dict(n=0,mean=None,std=None)
    from examples.mpc.fingertips.test.run_cso_contact_guidance import timing_cell
    assert timing_cell(r['models']['lcp']['execution_time_s_success_only'])=='--'
