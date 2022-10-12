# script to get ADO rate from germline SNPs
# run in mosaic-custom conda environment
# input should have been annotated with matched bulk normal ('AF-matched_bulk_normal' needs to be in sample.dna.col_attrs.keys())




import mosaic.io as mio
import pandas as pd
import numpy as np
from pathlib import Path
from tea.plots import plot_snv_clone
import json
import argparse, sys

def main(args):
    # ----- io ----- 
    input_h5 = Path(args.input_h5)
    if args.sample_name is None:
        sample_name = input_h5.name.split('.')[0]
    else:
        sample_name = args.sample_name

    # parameters for identifying germline HET SNPs
    sc_mean_AF_lower_bound = args.sc_mean_AF_lower_bound * 100 # times 100 for Tapestri data 
    sc_mean_AF_upper_bound = args.sc_mean_AF_upper_bound * 100
    bulk_AF_lower_bound = args.bulk_AF_lower_bound

    output_dir = Path(args.output_dir)
    # -----------------

    sample = mio.load(str(input_h5), raw = False)
    sample.dna.genotype_variants(
        min_dp = 8,
        min_alt_read = 3,
    )
    assert('AF-matched_bulk_normal' in sample.dna.col_attrs.keys())
    germline_snps = pd.DataFrame(index = sample.dna.ids()[
    ~np.isnan(sample.dna.col_attrs['AF-matched_bulk_normal'])
    ])
    germline_snps['AF-matched_bulk_normal'] = sample.dna.col_attrs['AF-matched_bulk_normal'][
        ~np.isnan(sample.dna.col_attrs['AF-matched_bulk_normal'])
        ]
    
    # ----- calcualte parameters for filtering germline HET SNPs -----
    cell_thpt = sample.dna.shape[0]
    germline_snps['sc_mut(AF>0)_prev'] = germline_snps.index.map(
        lambda x: float((sample.dna.get_attribute('AF', features = [x]) > 0).sum(axis=0)[0] / cell_thpt)
    )
    germline_snps['sc_mean_AF(positive cells only)'] = germline_snps.index.map(
        lambda x: sample.dna.get_attribute('AF', features = [x]).replace(0, np.NaN).mean()[0]
    )

    germline_snps['sc_detec(DP>0)_prev'] = germline_snps.index.map(
        lambda x: float((sample.dna.get_attribute('DP', features = [x]) > 0).sum(axis=0)[0] / cell_thpt)
    )

    germline_snps['ADO'] = germline_snps.index.map(
        lambda x: sample.dna.get_attribute('AF', features = [x]).isin([0, 100]).sum(axis=0)[0] / cell_thpt
    )

    # ----- filter for germline HET SNPs -----
    filtered_g_snps = germline_snps[
        (germline_snps['sc_mean_AF(positive cells only)'] > sc_mean_AF_lower_bound) & 
        (germline_snps['sc_mean_AF(positive cells only)'] < sc_mean_AF_upper_bound) & 
        (germline_snps['AF-matched_bulk_normal'] > bulk_AF_lower_bound) 
        # & (germline_snps['AF-matched_bulk_normal'] < 0.8) * (germline_snps['sc_detec(DP>0)_prev'] >= 0.75)
    ]

    # ----- write outputs -----
    filtered_g_snps.to_csv(output_dir / f'{sample_name}-germline_SNPs_info.csv', index = True, header = True)
    ado_stats = {}
    ado_stats['num_germline_snps_identified'] = filtered_g_snps.shape[0]
    ado_stats['mean_ADO'] = filtered_g_snps['ADO'].mean()
    with open(output_dir / f"{sample_name}-ado_stats.json", "w") as out_json:
        json.dump(ado_stats, out_json)
    
    # make sc-AF heatmap for sanity check
    sc_af_heatmap = plot_snv_clone(
        sample_obj= sample, 
        sample_name = sample_name,
        story_topic = 'germline_SNPs',
        voi = filtered_g_snps.index.tolist(),
        attribute = 'AF_MISSING',
        vars_to_sort_by = filtered_g_snps.index.tolist(),
        barcode_sort_method='hier',
    )
    sc_af_heatmap.write_image(str(output_dir / f'{sample_name}-germline_SNP_sc_AF_heatmap.png'), format="png", width=30 * filtered_g_snps.shape[0], height=0.5 * sample.dna.shape[0], scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_name', type=str, help='sample name', default=None)
    parser.add_argument('--input_h5', type=str, help='input h5 file')
    parser.add_argument('--sc_mean_AF_lower_bound', type=float, help='For identifying germline SNPs: sc_mean_AF_lower_bound (AF=0 not considered)', default=0.2,)
    parser.add_argument('--sc_mean_AF_upper_bound', type=float, help='sc_mean_AF_upper_bound (AF=0 not considered)', default=0.8,)
    parser.add_argument('--bulk_AF_lower_bound', type=float, help='For identifying germline SNPs: bulk_AF_lower_bound', default=0.2)
    parser.add_argument('--output_dir', type=str, help='output directory to save CRAVAT results. Default to the parent directory of input H5.', default = None)

    args = parser.parse_args(None if sys.argv[1:] else ['-h'])

    main(args)