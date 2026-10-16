"""Compare the scored point sequence with the material map actually committed."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();results={}
    paths=list((args.root/'comfree').glob('*/templates.npz'))
    paths+=list(args.root.glob('direct_*/templates.npz'))
    for path in paths:
        saved=np.load(path);rows=[]
        consistent='proposed_dense' in saved
        for index,window in enumerate(saved['window']):
            if not saved['updated'][index]:continue
            scored=saved['proposed_dense'][index] if consistent else np.repeat(saved['proposed'][index],6,axis=0)
            error=1000*np.linalg.norm(saved['template'][index]-scored,axis=-1)[saved['mask'][index]>0]
            maximum=float(np.max(error,initial=0.))
            if consistent:assert maximum<1e-4,(path,window,maximum)
            rows.append(dict(window=int(window),time_s=float(window*.03),max_error_mm=maximum,
                mean_error_mm=float(error.mean()) if error.size else 0.))
        results[str(path.parent.relative_to(args.root))]=dict(commit_consistent=consistent,updates=len(rows),
            max_error_mm=max((r['max_error_mm'] for r in rows),default=0.),rows=rows)
    (args.root/'template_commit_discrepancy.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in value.items() if k!='rows'} for key,value in results.items()},indent=2))


if __name__=='__main__':main()
