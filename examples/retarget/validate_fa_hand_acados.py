"""Check the condensed acados QP against analytic solutions and box KKT."""
import json
from pathlib import Path
import numpy as np
from examples.retarget.fa_hand_acados import CondensedHandQP


def main():
    rng=np.random.default_rng(20261016);qp=CondensedHandQP(130);rows=[]
    for index in range(3):
        a=rng.normal(size=(130,130));H=a.T@a/130+.2*np.eye(130);g=rng.normal(size=130)
        exact=np.linalg.solve(H,-g)
        for trust in (100.,1.):
            step,status=qp.solve(g,H,trust)
            residual=step-np.clip(step-(H@step+g),-trust,trust)
            row=dict(index=index,trust=trust,status=status,projected_KKT_max_error=float(np.max(abs(residual))),
                analytic_solution_max_error=float(np.max(abs(step-exact))) if trust==100. else None)
            assert status==0,row
            assert row['projected_KKT_max_error']<1e-6,row
            if trust==100.:assert row['analytic_solution_max_error']<1e-6,row
            rows.append(row)
    root=Path('outputs/fa_matched_ablation_20261016')
    (root/'condensed_qp_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
