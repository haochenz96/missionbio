# script to plot barcode performance used for default Tapestri pipeline cell-calling. This was derived from the cellfinder.py script of the default pipeline.
# run in mosaic-custom conda environment
# input should be {sample_name}.cellfinder.barcode.distribution.tsv (unfiltered read count matrix), which is the output from part1 of the default Tapestri pipeline

import argparse
import logging
import time
import pandas as pd
from pathlib import Path
import glob
import plotly.express as px

def make_cli_parser():
    cli_parser = argparse.ArgumentParser(description=__doc__)
    cli_parser.add_argument(
        '--rc_tsv_path', 
        help='Path to a the {sample_name}.cellfinder.barcode.distribution.tsv (unfiltered read count matrix)'
    )
    cli_parser.add_argument(
        '--minor_threshold', default = 0.2,
        help='percent of amplicons to keep'
    )
    cli_parser.add_argument(
        '--output_dir', 
        help='output directory for histogram plots'
    )
    cli_parser.add_argument(
        '--sample_name', default = None,
        help='sample name to use for file naming purpose'
    )
    return cli_parser

def main(argv=None):

    cli_parser = make_cli_parser()
    args = cli_parser.parse_args(argv)

    # ----- get inputs -----
    minor_threshold = float(args.minor_threshold)
    if not 0 <= minor_threshold <= 1:
        raise ValueError("minor_threshold must be between 0 and 1")

    output_dir = Path(args.output_dir)
    print(f"[INFO] ---- output directory: {output_dir}")

    # retrieve unfiltered RC TSV
    unfiltered_rc_tsv = Path(args.rc_tsv_path)
    print(f"[INFO] ---- unfiltered RC TSV: {unfiltered_rc_tsv}")
    # unfitlered_rc_df = read_tsv(unfiltered_rc_tsv)

    sample_name = args.sample_name
    if sample_name is None:
        sample_name = unfiltered_rc_tsv.name.split('.tube1.cellfinder')[0] # by default, infer from TSV file's basename
    print(f"[INFO] ---- sample name: {sample_name}")

    # filter based on MB cellfilder.py
    # ----- step1: filter out barcodes based on a total read cutoff ----- 
    X_1 = get_candidate_barcodes(unfiltered_rc_tsv) # output: X1 matrix
    print(f"[INFO] ---- finished STEP1")
    print(f"[INFO] ---- X1: {X_1.shape}")
    X_1.to_csv(str(output_dir / f"{sample_name}_X1.tsv"), sep='\t')

    # ----- step2: filter out amplicons based on a total read cutoff -----
    rc_thres = calculate_threshold(X_1, minor_threshold) # thres = 0.2 * mean(X1), by default
    print(f"[INFO] ---- mean rc threshold is set to {rc_thres}")
    X_2 = remove_amplicons(X_1, rc_thres)
    print(f"[INFO] ---- finished STEP2 and excluded low-performing amplicons in cellfinder calculation")
    print(f"[INFO] ---- remaining number of amplicons: {X_2.shape[1]}")
    # ----- step 2.5 @HZ: make the histogram
    plot_barcode_amplicon_performance_distribution(X_2, minor_threshold, rc_thres, output_dir, sample_name=sample_name)
    print(f"[INFO] ---- finished plotting the histogram")
    # ----- step3: find cells -----
    # (ncells, cellBarcodes) = get_cell_count(X_2, rc_thres) # output: X3 matrix

def plot_barcode_amplicon_performance_distribution(df, minor_threshold, rc_thres, out_dir, sample_name=None):
    """
    
    first get rid of failing amplicons (amplicon with mean_rc < threshold * mean(df))

    Plot distribution of single cells in terms of amplicon performance 
    (%amplicons with rc >= threshold * mean(df))
    """

    if sample_name is None:
        sample_name = 'test sample'

    df = df
    total_cell_num = len(df.index)
    ntargets = len(df.columns)
    df["percentthres"] = df.apply(
        func=lambda row: count_values_in_range(row, rc_thres, ntargets), axis=1)
    ncalled_cells = len(df[df.percentthres > 80].index)
    fig = px.histogram(df, x="percentthres", nbins=100)
    fig.update_xaxes(
        title = f'%amplicons with mean_rc >= {rc_thres}',
    )
    fig.update_layout(
        title=f"""
        {sample_name} single-cell amplicon performance distribution<br>
        <sup>candidates: {total_cell_num} cells</sup>
        <sup>cells called: {ncalled_cells} cells</sup>
        """,
    )
    fig.add_vline(
        80, 
        line_color='red', 
        line_width=2, row=1, col=1, opacity=0.2,
        annotation_text=f"default cellfinder cutoff = 80%",
        annotation_font_size=10,
        annotation_font_color="red",
    )
    
    fig.write_image(str(out_dir / f"{sample_name}_single-cell_amplicon_performance_distribution.png"), 
                    width=1000, height=500, scale=2)

# ----- below are functions from MB's cell_utils.py ----- 

# step1: filter barcodes based on a total read cutoff
def get_candidate_barcodes(unfiltered_rc_tsv, total_rc_cutoff_multiplier = 8):
    '''
    Filters an all-barcode matrix for candidate barcodes based on a total read count cutoff, which is default to be 8 * number of amplicons

    Parameters
    ----------
    unfiltered_rc_tsv: str
        the unfiltered read count tsv file
    total_rc_cutoff_multiplier: int
        total read count cutoff multiplier (for a barcode to be considered, it needs to have multiplier * number of amplicons), default to be 8

    Returns
    -------
    pandas.DataFrame
        matrix with candidate barcodes

    '''
    # input as path to unfiltered RC matrix
    # df = read_tsv(unfiltered_rc_tsv)
    with pd.read_csv(unfiltered_rc_tsv, sep='\t', index_col=0, header=0) as df:
        num_amplicons = len(df.columns)
        threshold = total_rc_cutoff_multiplier * num_amplicons
        df_candidate = df[df.sum(axis=1) > threshold]
    return df_candidate

# step2
def remove_amplicons(df, thres):
    pd.options.mode.chained_assignment = None
    df = df[df.columns[df.mean(axis=0) > thres]]
    df['sum'] = df.sum(axis=1)
    df = df[df['sum'] > thres] # @HZ 06/07/2022- i don't understand why this filter is put here. This shouldn't filter out anything.
    df = df.drop('sum', axis=1)
    return df

# step3
def get_cell_count(df, thres):
    ntargets = len(df.columns)
    df["percentthres"] = df.apply(
        func=lambda row: count_values_in_range(row, thres, ntargets), axis=1)
    df = df[df.percentthres > 80]
    ncells = len(df.index)
    df = df.drop('percentthres', axis=1)
    return (ncells, df.index)

# --- utility functions ---
# @HZ: this function in the default pipeline is incredibly wrong
def read_tsv(inputtsv):
    df = pd.read_csv(inputtsv, sep='\t')
    mod_col = df.columns[1:]
    mod_df = df.iloc[:, :-1] # <--- This is excluding the last amplicon
    mod_df.columns = mod_col
    df = mod_df
    df.index.name = 'cell_barcode'
    return df

def calculate_threshold(df, threshold):

    """Convert percentage threshold into absolute value

    Uses mean of the data as a reference point.

    Parameters
    ----------
    df: pd.DataFrame
        reference data
    threshold: float
        threshold represented as percentage

    Returns
    -------
    float
        absolute threshold
    """
    return threshold * df.mean().mean()

def count_values_in_range(series, range_min, ntargets):
    return series.ge(range_min).sum() * 100 / ntargets

# ----- execution ----- 
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exiting abnormally due to {}".format(str(e)))