"""
Module to read and write h5 files
"""

import os
import time
import warnings
from copy import deepcopy
from os import path

import h5py
import numpy as np
import pandas as pd
from missionbio.h5.constants import (
    BARCODE,
    DATE_CREATED,
    DNA_ASSAY,
    DNA_READ_COUNTS_ASSAY,
    ID,
    METADATA,
    PROTEIN_ASSAY,
    SAMPLE,
    SDK_VERSION,
)
from missionbio.h5.data import H5Reader, H5Writer
from missionbio.h5.merge import merge_assays, merge_samples

from missionbio.mosaic.cnv import Cnv
from missionbio.mosaic.constants import READS
from missionbio.mosaic.dna import Dna
from missionbio.mosaic.protein import Protein
from missionbio.mosaic.sample import Sample


def _init_from_assay(assay, assay_type):
    """
    Initialze using missionbio.h5.assay.Assay object.
    Determines the shape if missing.

    Parameters
    ----------
    assay : missionbio.h5.assay.Assay

    assay_type : class
        One of the following:
        - :class:`missionbio.mosaic.dna.Dna`
        - :class:`missionbio.mosaic.cnv,Cnv`
        - :class:`missionbio.mosaic.protein.Protein`

    Returns
    -------
    assay_type
    """
    new_assay = assay_type(name=assay.name,
                           metadata=assay.metadata,
                           layers=assay.layers,
                           row_attrs=assay.row_attrs,
                           col_attrs=assay.col_attrs)

    for _, layer in assay.layers.items():
        new_assay.shape = layer.shape
        break
    else:
        new_assay.shape = (0, 0)

    return new_assay


def _loom_to_h5(loom_file, tsv_file=None, vcf_header=None):
    """
    Reogranizes the data within the loom file
    to the standard h5 file format.

    This function replaces the existing loom file.
    Make sure to make a copy.

    Parameters
    ----------
    loom_file : str
        Path to the loom file

    tsv_file : str
        Path to the cell read coount distribution file

    vcf_header : str
        Path to the vcf header file. If 'col_attrs/barcode'
        exists in the looom file, this file will be ignored.
        If 'col_attrs/barcode' does not exist, then
        this file is required.

    Raises
    ------
    Exception
        If the loom file does not contain the
        'col_attrs/barcode' dataset and the
        vcf_header file is not provided.
    """

    print(f'Reading {loom_file}')
    with h5py.File(loom_file, 'a') as f:
        if 'barcode' not in f['col_attrs'].keys():
            if vcf_header is None:
                raise Exception('"col_attrs/barcode" not found in the loom file. "vcf_header" must be passed to create the h5 file.')

            with open(vcf_header, 'r') as vf:
                barcodes = vf.read().strip().split('\t')
                f['col_attrs/barcode'] = np.array(barcodes, dtype=np.bytes_)
        elif vcf_header is not None:
            warnings.warn(f'"col_attrs/barcode" found in the loom file. Given "vcf_header" file - {vcf_header} is being ignored.')

        print('Reorganizing DNA data')
        f['assays/dna_variants/ca'] = f['row_attrs']
        f['assays/dna_variants/ra'] = f['col_attrs']
        ad = f['layers/AD'][()].T
        dp = f['layers/DP'][()].T
        gq = f['layers/GQ'][()].T
        ngt = f['matrix'][()].T

        af = np.zeros(ad.shape)
        af[dp != 0] = ad[dp != 0] / dp[dp != 0]
        af = af * 100

        f['assays/dna_variants/layers/AF'] = af
        f['assays/dna_variants/layers/DP'] = dp
        f['assays/dna_variants/layers/NGT'] = ngt
        f['assays/dna_variants/layers/GQ'] = gq

        f['assays/dna_variants/metadata/sample_name'] = np.array(['cnv_run']).astype(np.bytes_)
        f['assays/dna_variants/ra/sample_name'] = np.array(['cnv_run'] * af.shape[0]).astype(np.bytes_)

        f['metadata/date_created'] = np.bytes_(b'2021-02-19')
        f['metadata/sdk_version'] = np.bytes_(b'3.2.0c')

        del f['col_attrs']
        del f['row_attrs']
        del f['layers']
        del f['matrix']
        del f['row_graphs']
        del f['col_graphs']

        if tsv_file is not None:
            df = pd.read_csv(tsv_file, sep='\t', index_col=0)

            print('Reorganizing CNV data')

            f['assays/dna_read_counts/layers/read_counts'] = df.values
            f['assays/dna_read_counts/ra/barcode'] = df.index.values.astype(np.bytes_)
            f['assays/dna_read_counts/ca/id'] = df.columns.values.astype(np.bytes_)
            f['assays/dna_read_counts/metadata/sample_name'] = np.array(['cnv_run']).astype(np.bytes_)
            f['assays/dna_read_counts/ra/sample_name'] = np.array(['cnv_run'] * df.shape[0]).astype(np.bytes_)

    h5_file = '.h5'.join(loom_file.rsplit('.loom', 1))
    os.rename(loom_file, h5_file)

    print('File structure converted successfully. The extension was renamed from loom to h5')
    print('Reorganizing barcode order')

    sample = load(h5_file)
    sample.dna.normalize_barcodes()
    sample.dna.select_rows(sample.dna.barcodes().argsort())

    if sample.cnv is not None:
        sample.cnv.normalize_barcodes()
        sample.cnv.select_rows(sample.cnv.barcodes().argsort())

    os.remove(h5_file)
    save(sample, h5_file)

    print('Successfully restructured the data.')


def _is_supported(h5file):
    """
    Check if the file can be read.

    Parameters
    ----------
    h5file : str
        The path to the .h5 file.

    Returns
    -------
    bool
        Whether the h5 files is supported
        by missionbio.h5.data.H5Reader or not.
    """

    with h5py.File(h5file, 'r+') as file:
        if METADATA not in file:
            grp = file.create_group(METADATA)
            grp[DATE_CREATED] = np.bytes_('1900-01-01')
            grp[SDK_VERSION] = np.bytes_('0')

        version = file[METADATA][SDK_VERSION][()].astype(str)
        if version[-1] != 'c' and int(version.split('.')[0]) < 3:
            return False

    return True


def _update_file(h5file):
    """
    Update the old .h5 files to the latest format.

    Parameters
    ----------
    h5file : str
        The path to the .h5 file.
    """

    if _is_supported(h5file):
        return

    print(f'Modifying {h5file} in place.')

    H5_ASSAY_NAMES = {
        'dna': DNA_ASSAY,
        'cnv': DNA_READ_COUNTS_ASSAY,
        'protein': PROTEIN_ASSAY,
    }

    with h5py.File(h5file, 'r+') as file:
        version = file[f'{METADATA}/{SDK_VERSION}'][()].astype(str)
        del file[METADATA][SDK_VERSION]
        version = np.bytes_(version + '.c')
        file.create_dataset(f'{METADATA}/{SDK_VERSION}', data=version)

        for assay_type, new_assay_type in zip(['assays', 'raw_counts'], ['assays', 'all_barcodes']):
            if assay_type in file.keys():
                for assay in file[assay_type].keys():
                    if assay == 'dna':
                        dp = file[f'{assay_type}/dna/layers/DP'][()]
                        ad = file[f'{assay_type}/dna/layers/AD'][()]
                        file[f'{assay_type}/dna/layers/AF'] = 100 * np.divide(ad, dp, where=(dp != 0))

                    if assay in H5_ASSAY_NAMES:
                        new_name = H5_ASSAY_NAMES[assay]
                        file[f'{new_assay_type}/{new_name}'] = file[assay_type][assay]
                        del file[assay_type][assay]

                        name = file[f'{new_assay_type}/{new_name}/{METADATA}/{SAMPLE}'][()].astype(str)
                        del file[f'{new_assay_type}/{new_name}/{METADATA}/{SAMPLE}']
                        file.create_dataset(f'{new_assay_type}/{new_name}/{METADATA}/{SAMPLE}', data=[bytes(name, 'utf-8')])

                        keys = file[f'{new_assay_type}/{new_name}/ra/'].keys()
                        num_bars = len(file[f'{new_assay_type}/{new_name}/ra/barcode'][()])
                        if SAMPLE in keys:
                            del file[f'{new_assay_type}/{new_name}/ra/{SAMPLE}']
                        file.create_dataset(f'{new_assay_type}/{new_name}/ra/{SAMPLE}', data=np.array([bytes(name, 'utf-8')] * num_bars))


def merge_files(folderpath, name, update=False):
    """
    Convert the legacy files to .h5.

    This is meant to be used with legacy versions of the pipeline to generate new .h5 files.

    The files should have the following names:
        - DNA .h5 file : dna.h5
        - Protein .h5 file : protein.h5
        - DNA raw counts : {prefix-}cellfinder.distribution.tsv
        - Protein raw counts: {prefix-}counts.tsv

    If the .h5 file has raw counts, the provided files are ignored.

    Parameters
    ----------
    folderpath : str
        path of the folder which contains the
        appropriately named files

    name : str
        the name that is to be given to the new h5 file

    update :  bool
        Whether the h5 file should be updated
        from the legacy version to a currently supported
        version of the h5 format.

    Raises
    ------
    KeyError
        If "dna.h5" is not availble in the folder.

    Exception
        If the h5 file format is not supported.
    """

    # Potential assays to be filled if found, none by default
    dna = None
    cnv = None
    protein = None
    cnv_raw = None
    protein_raw = None

    # Find the files in the folder
    files = [f for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f))]

    # Return if .h5 file already exists
    if name + '.h5' in files:
        return

    assay_names = ['dna', 'protein', 'cnv_raw', 'protein_raw']
    assay_extension = ['dna.h5', 'protein.h5', 'cellfinder.distribution.tsv', '-counts.tsv', 'none']

    data_files = {}
    for i in range(len(assay_names)):
        for file in os.listdir(folderpath):
            if file.endswith(assay_extension[i]):
                data_files[assay_names[i]] = folderpath + '/' + file
            elif assay_names[i] not in data_files:
                data_files[assay_names[i]] = ''

    print(data_files)

    def check_file(h5file):
        if update:
            _update_file(h5file)
        elif not _is_supported(h5file):
            raise Exception(f"The version of .h5 file {h5file} is not supported. Update the file to a supported format by passing update=True.")

    # Look for the DNA assay
    if path.exists(data_files['dna']):
        check_file(data_files['dna'])

        with H5Reader(data_files['dna']) as reader:
            print('Loading DNA')
            dna = reader.read(DNA_ASSAY)
            cnv = reader.read(DNA_READ_COUNTS_ASSAY)

            raw_count_list = reader.raw_counts()
            if DNA_READ_COUNTS_ASSAY in raw_count_list:
                cnv_raw = reader.read_raw_counts(DNA_READ_COUNTS_ASSAY)
            elif path.exists(data_files['cnv_raw']):
                cnv_raw = _cnv_raw_counts(data_files['cnv_raw'])
    else:
        raise KeyError('Include the dna.h5 file at a minimum in the folder ' + folderpath)

    # Look for the Protein assay
    if path.exists(data_files['protein']):
        check_file(data_files['protein'])

        with H5Reader(data_files['protein']) as reader:
            print('Loading Protein')
            protein = reader.read(PROTEIN_ASSAY)
            raw_count_list = reader.raw_counts()

            if PROTEIN_ASSAY in raw_count_list:
                protein_raw = reader.read_raw_counts(PROTEIN_ASSAY)
            elif path.exists(data_files['protein_raw']):
                protein_raw = _protein_raw_counts(data_files['protein_raw'])

    # Prepare and merge the assays
    assays = [dna, cnv, protein]
    raw_counts = [cnv_raw, protein_raw]

    assays = [assay for assay in assays if assay is not None]
    raw_counts = [raw_count for raw_count in raw_counts if raw_count is not None]

    for assay in assays:
        assay.normalize_barcodes()
        assay.add_metadata(SAMPLE, [name])
        assay.add_row_attr(SAMPLE, np.array([name] * assay.shape[0]))

    for raw_count in raw_counts:
        raw_count.normalize_barcodes()

    assays = merge_assays(*assays)

    # Write to .h5
    with H5Writer(f'{folderpath}/' + name + '.h5', mode='w') as w:
        print('Saving')
        for assay in assays:
            w.write(assay)

        for raw_count in raw_counts:
            w.write_raw_counts(raw_count)


def _cnv_raw_counts(file_path):
    """
    Loading the DNA raw counts from a .tsv file.

    Some .h5 files do not contain raw counts,
    hence this method is needed.

    Parameters
    ----------
    file_path : str
        The path to the cellfinder.barcode.distribution.tsv
        file from the DNA run.

    Returns
    -------
    missionbio.mosaic.cnv.Cnv
    """

    df = pd.read_csv(file_path, sep='\t')
    cols = df.columns[1:]
    df = df.iloc[:, :-1]
    df.columns = cols

    return Cnv(name='cnv',
               metadata={SAMPLE: ['raw_counts']},
               layers={READS: df.to_numpy(dtype='uint32')},
               row_attrs={BARCODE: np.array(df.index)},
               col_attrs={ID: np.array(df.columns)})


def _protein_raw_counts(file_path):
    """
    Loading the protein raw counts from a .tsv file.

    Some .h5 files do not contain raw counts,
    hence this method is needed.

    Parameters
    ----------
    file_path : str
            The path to the XXXX_protein-counts.tsv
            file from the Protein run.

    Returns
    -------
    missionbio.mosaic.protein.Protein
    """

    df = pd.read_csv(file_path, index_col=[0, 1], sep='\t')
    df = df.iloc[:, 0]
    df = df.unstack(fill_value=0)

    return Protein(name='protein',
                   metadata={SAMPLE: ['raw_counts']},
                   layers={READS: df.to_numpy(dtype='uint32')},
                   row_attrs={BARCODE: np.array(df.index)},
                   col_attrs={ID: np.array(df.columns)})


def load(filepath, name=None, raw=False, update=False, apply_filter=False, whitelist=None):
    """
    Loading the .h5 file with one or more assays.

    This is the preferred way of loading .h5 files.

    It directly returns a `Sample` object, which
    contains all the assays. Those assays that were
    not present are stored as `None`.

    Parameters
    ----------
    filepath : str
        The path to the .h5 multi-omics file.
    name : str
        Name to be given to the sample. If `None`,
        then the dna sample name from the h5 file
        is used.
    raw : bool
        Whether the raw counts are to be loaded.
    update : bool
        Whether the file is to be updated
        if the .h5 format is not supported.
    apply_filter : True
        Whether to load only the filtered dna variants.
    whitelist : list-like
        The specific dna variants to load.

    Returns
    -------
    missionbio.mosaic.sample.Sample

    Raises
    ------
    Exception
        When the h5 file format is not supported.
    """

    # Assays to be loaded are None unless filled
    dna = None
    cnv = None
    protein = None
    cnv_raw = None
    protein_raw = None

    if update:
        _update_file(filepath)
    elif not _is_supported(filepath):
        raise Exception(f"The version of the .h5 file {filepath} is not supported. Update the file to a supported format by passing update=True.")

    start_time = time.time()
    print(f'Loading, {filepath}')
    with H5Reader(filepath) as reader:
        assay_list = reader.assays()
        raw_count_list = reader.raw_counts()

        if DNA_ASSAY in assay_list:
            dna = _init_from_assay(reader.read(DNA_ASSAY, apply_filter=apply_filter, whitelist=whitelist), Dna)
        if DNA_READ_COUNTS_ASSAY in assay_list:
            cnv = _init_from_assay(reader.read(DNA_READ_COUNTS_ASSAY), Cnv)
        if PROTEIN_ASSAY in assay_list:
            protein = _init_from_assay(reader.read(PROTEIN_ASSAY), Protein)

        if raw:
            if DNA_READ_COUNTS_ASSAY in raw_count_list:
                cnv_raw = _init_from_assay(reader.read_raw_counts(DNA_READ_COUNTS_ASSAY), Cnv)
            if PROTEIN_ASSAY in raw_count_list:
                protein_raw = _init_from_assay(reader.read_raw_counts(PROTEIN_ASSAY), Protein)

    print(f'Loaded in {(time.time() - start_time):.1f}s.')

    return Sample(name=name,
                  dna=dna,
                  cnv=cnv,
                  protein=protein,
                  cnv_raw=cnv_raw,
                  protein_raw=protein_raw)


def save(sample, path, raw=False):
    """
    Save the analyzed sample as .h5.

    Parameters
    ----------
    sample : :class:`missionbio.mosaic.sample.Sample`
        The sample to be saved.
    path : str
        The path to save it to.
    raw : bool
        Whether to save the raw counts
        of the assays or not.
    """

    with H5Writer(path, mode='w-') as w:
        assays = [sample.protein, sample.dna, sample.cnv]
        for assay in assays:
            if assay is not None:
                w.write(assay)

        if raw:
            raw_counts = [sample.protein_raw, sample.cnv_raw]

            for raw_count in raw_counts:
                if raw_count is not None:
                    w.write_raw_counts(raw_count)


def merge(samples):
    """
    Merge multiple samples into one.

    3 bases are appended to the barcodes in each sample
    to avoid the collision of barcodes across samples.

    Parameters
    ----------
    samples : list
        The list of samples to merge.

    Returns
    -------
    combo : :class:`missionbio.mosaic.sample.Sample`
        A new sample with 'sample_id' added to the
        row_attrs showing the sample that the barcode
        belongs to.
    """
    assay_names = ['dna', 'protein', 'cnv',
                   'cnv_raw', 'protein_raw']

    assay_types = [Dna, Protein, Cnv,
                   Cnv, Protein]

    samples = deepcopy(samples)

    # Create 3-base barcode prefix for up to 64 samples so that barcodes from different samples don't collide
    bases = 'ACGT'
    sampleids = list(np.concatenate([[[bases[i] + bases[j] + bases[k] for k in range(4)] for j in range(4)] for i in range(4)]).flat)

    combo = Sample()

    for assay_name, assay_type in zip(assay_names, assay_types):

        assays = [sample.__dict__[assay_name] for sample in samples]

        if None in assays:
            combo.__dict__[assay_name] = None
            continue

        print(f'Combining {assay_name}')

        # Append the sample prefix to the barcodes
        for i in range(len(assays)):
            assays[i].row_attrs[BARCODE] = [sampleids[i] + barcode for barcode in assays[i].row_attrs[BARCODE]]

        # Merge samples
        assay = _init_from_assay(merge_samples(*assays, inplace=True), assay_type)
        combo.__dict__[assay_name] = assay

    combo._original_dna = combo.dna
    combo._original_cnv = combo.cnv
    combo._original_protein = combo.protein

    return combo
