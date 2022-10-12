import json
import glob
from pathlib import Path
import pandas as pd

results_dir = Path('/home/zhangh5/work/Tapestri_batch1/analysis/technical/ADO/ADO_calc_results')
js = glob.glob(
    str(results_dir / '*.json')
    )

summary_df = pd.DataFrame(columns = ['num_germline_snps_identified', 'mean_ADO'])
summary_df.index.name = 'sample_code'
for ji in js:
    sample_name = Path(ji).name.split('-ado')[0]
    with open(ji, 'r') as f:
        d = json.load(f)
        rec = pd.Series(d)
    summary_df.loc[sample_name] = rec

summary_df['num_germline_snps_identified'] = summary_df['num_germline_snps_identified'].astype(int)
summary_df.to_csv(str(results_dir / 'all_samples_ADO_summary.csv'), index = True, header = True)
