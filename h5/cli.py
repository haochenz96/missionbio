"""
Command line interface for h5 data manipulation methods.
"""

import json
import logging
import os
import shutil
import sys
from contextlib import ExitStack

import click
import numpy as np
from missionbio.cli import tapestri_command

from missionbio.h5.constants import BARCODE, ID
from missionbio.h5.create import create_cnv_assay, create_dna_assay, create_protein_assay
from missionbio.h5.data import H5Reader, H5Writer
from missionbio.h5.exceptions import ValidationError
from missionbio.h5.filter import FilterConfig, add_variant_stats, variant_filtering_options
from missionbio.h5.merge import merge_assay_files, merge_samples
from missionbio.h5.utils import find_common_keys, is_sorted

log = logging.getLogger(__name__)


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    root_logger.addHandler(handler)


@tapestri_command.group()
def h5():
    setup_logging()


@h5.group()
def create():
    pass


@create.command()
@click.option(
    "--vcf", "vcf_file", required=True, type=click.Path(exists=True), help="path to the vcf file"
)
@click.option(
    "--read-counts",
    "read_counts_file",
    required=True,
    type=click.Path(exists=True),
    help="path to the read counts tsv file",
)
@click.option(
    "--metadata",
    "metadata_file",
    type=click.Path(exists=True),
    help="path to the metadata.json file",
)
@click.option(
    "--output", "output_file", required=True, type=click.Path(), help="name of the output file"
)
def dna(vcf_file, read_counts_file, metadata_file, output_file):
    if metadata_file is not None:
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    dna = create_dna_assay(vcf_file, metadata)
    cnv = create_cnv_assay(read_counts_file, metadata)

    with H5Writer(output_file) as writer:
        writer.write(dna)
        writer.write(cnv)


@create.command()
@click.option(
    "--read-counts",
    "read_counts_file",
    required=True,
    type=click.Path(exists=True),
    help="path to the read counts tsv file",
)
@click.option(
    "--metadata",
    "metadata_file",
    type=click.Path(exists=True),
    help="path to the metadata.json file",
)
@click.option(
    "--output", "output_file", required=True, type=click.Path(), help="name of the output file"
)
def protein(read_counts_file, metadata_file, output_file):
    if metadata_file is not None:
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    protein_assay = create_protein_assay(read_counts_file, metadata)

    with H5Writer(output_file) as writer:
        writer.write(protein_assay)


@h5.group()
def merge():
    pass


@merge.command(
    help="\n".join(
        [
            "Merge assays from SRC into DST",
            "",
            "SRC are two or more h5 files to merge.",
            "DST is the name of the file to create.",
            "If DST file already exists, it will be overwritten.",
        ]
    )
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Rename samples to the name from the first file. "
    "Requires that all files contain a single sample",
)
@click.argument("src", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.argument("dst", nargs=1, type=click.Path(dir_okay=False, exists=False))
def assays(src, dst, force):
    if not src or len(src) < 2:
        click.echo("At least two SRC and one DST filenames are required", err=True)
        sys.exit(1)

    if not merge_assay_files(src, dst, force):
        sys.exit(1)


@merge.command(
    help="\n".join(
        [
            "Merge samples from SRC into DST",
            "",
            "SRC are two or more h5 files to merge.",
            "DST is the name of the file to create.",
            "If DST file already exists, it will be overwritten.",
        ]
    )
)
@click.argument("src", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.argument("dst", nargs=1, type=click.Path(dir_okay=False, exists=False))
def samples(src, dst):
    if not src or len(src) < 2:
        click.echo("At least two SRC and one DST filenames are required", err=True)
        sys.exit(1)

    with ExitStack() as stack:
        readers = []
        for fn in src:
            try:
                readers.append(stack.enter_context(H5Reader(fn)))
            except ValidationError as err:
                log.error(f"Could not load {fn}. {err}")
                sys.exit(1)

        writer = stack.enter_context(H5Writer(dst, mode="w"))

        common_assays = find_common_keys(
            (reader.assays() for reader in readers),
            warning="Some assays are missing in {name} "
            "({missing_keys}). Only assays that "
            "are present in all input files will "
            "be merged",
            names=src,
        )
        for assay in common_assays:
            assays = []
            for reader in readers:
                log.info(f"Reading {assay} assay from {reader.filename}")
                assays.append(reader.read(assay))

            log.info("Merging assays")
            merged = merge_samples(*assays, inplace=True)

            log.info(f"Writing {merged} assay to {dst}")
            writer.write(merged)

        log.info("Done")


@h5.command()
@click.argument("filenames", nargs=-1)
def sort(filenames):
    for fn in filenames:
        tmp_filename = fn + ".tmp"
        try:
            log.info(f"Checking {fn}")
            with ExitStack() as stack:
                reader = stack.enter_context(H5Reader(fn))
                writer = stack.enter_context(H5Writer(tmp_filename, mode="w"))

                for assay_name in reader.assays():
                    log.info(f"Reading assay {assay_name}")
                    assay = reader.read(assay_name)

                    if not is_sorted(assay.row_attrs[BARCODE]):
                        log.info("sorting rows")
                        order = np.argsort(assay.row_attrs[BARCODE])
                        assay.select_rows(order)
                    else:
                        log.info("rows already sorted")

                    if not is_sorted(assay.col_attrs[ID]):
                        log.info("sorting columns")
                        order = np.argsort(assay.col_attrs[ID])
                        assay.select_columns(order)
                    else:
                        log.info("columns already sorted")

                    log.info(f"Writing {assay}")
                    writer.write(assay)

            shutil.move(tmp_filename, fn)
        finally:
            try:
                os.remove(tmp_filename)
            except Exception:
                pass


@h5.command(help="Rename samples in files inplace")
@click.option("--old-name", "old_names", required=True, multiple=True)
@click.option("--new-name", required=True)
@click.argument("filenames", nargs=-1)
def rename_sample(old_names, new_name, filenames):
    for fn in filenames:
        tmp_filename = fn + ".tmp"
        try:
            log.info(f"Checking {fn}")
            with ExitStack() as stack:
                reader = stack.enter_context(H5Reader(fn))
                writer = stack.enter_context(H5Writer(tmp_filename, mode="w"))

                for assay_name in reader.assays():
                    log.info(f"Reading assay {assay_name}")
                    assay = reader.read(assay_name)
                    log.info(f"Found samples {assay.samples()}")

                    to_rename = assay.samples() & set(old_names)
                    if len(to_rename) > 1:
                        log.error(
                            f"Multiple samples in {assay} match "
                            f"given old_names ({to_rename}). This"
                            f"is not supported"
                        )
                        return
                    elif len(to_rename) == 0:
                        log.warning(f"Nothing to rename in {assay}")
                    else:
                        (old_name,) = to_rename
                        assay.rename_sample(old_name, new_name)
                        log.info(f"Renamed {old_name} to {new_name} in {assay}")

                    log.info(f"Writing {assay}")
                    writer.write(assay)

            shutil.move(tmp_filename, fn)
        finally:
            try:
                os.remove(tmp_filename)
            except Exception:
                pass


@h5.command(
    help="""Compute filtering metrics for DNA assay with variant quality thresholds
    and store it in h5"""
)
@variant_filtering_options
def variant_stats(
    h5_file,
    gq_cutoff,
    dp_cutoff,
    af_cutoff,
    missing_cells_cutoff,
    missing_variants_cutoff,
    mutated_cells_cutoff,
):
    log.info(f"Collecting filtering stats for {h5_file}")
    config = FilterConfig(
        gq_cutoff,
        dp_cutoff,
        af_cutoff,
        missing_cells_cutoff,
        missing_variants_cutoff,
        mutated_cells_cutoff,
    )
    add_variant_stats(h5_file, config)
    log.info("Finished collecting filtering stats")


if __name__ == "__main__":
    tapestri_command()
