import logging
from multiprocessing.sharedctypes import Value
from typing import Any, Dict, Optional

import allel
import numpy as np
import pandas as pd
from pandas import Series

from h5.constants import (
    AF,
    ALT,
    CHROM,
    DEFAULT_SAMPLE,
    DNA_ASSAY,
    DP,
    GQ,
    ID,
    NGT,
    POS,
    QUAL,
    REF,
    RGQ,
    SAMPLE,
)
from h5.data import Assay

__all__ = ["create_dna_assay"]

log = logging.getLogger(__name__)

# @HZ 06/22/2022
# we want to have more control of which field to add to 'DNA' ASSAY
# for now, the input "custom_fields" can only be calldata (INFO) fields in the VCF
def create_dna_assay(
    vcf_file: str,
    custom_fields: Optional[list] = [],
    metadata: Optional[Dict[str, Any]] = None,
    quality_threshold: Optional[int] = None,
) -> Assay:
    """Create DNA assay from vcf data

    Args:
        vcf_file: path to the vcf file
        custom_fields: list of custom fields to add to the assay. Has to be a `FORMAT` field in the VCF.
        metadata: assay metadata
        quality_threshold: required quality to put variant in block1

    Returns:
        dna assay
    """
    if metadata is None:
        metadata = {}

    assay = Assay.create(DNA_ASSAY)

    log.info("Reading vcf file")
    vcf = VCFFile(vcf_file, quality_threshold=quality_threshold, custom_fields=custom_fields)

    log.info("Creating layers")
    assay.add_layer(NGT, vcf.create_ngt(dtype=np.int8).T)
    assay.add_layer(GQ, vcf.layer(VCFFile.GQ, remove_missing=True).T)
    assay.add_layer(RGQ, vcf.layer(VCFFile.RGQ, remove_missing=True).T)
    ad = vcf.layer(VCFFile.AD, remove_missing=True)[:, :, 1].T # alternate allele depth
    dp = vcf.layer(VCFFile.DP, remove_missing=True).T
    assay.add_layer(DP, dp)
    assay.add_layer(AF, compute_af(ad, dp))
    
    # add custom fields (e.g. TLOD)
    for field in custom_fields:
        vcf_field = f'calldata/{field}' # note the distinction between `field` and `vcf_field`- the latter is for retrieving the layer from the VCFFile instance
        # check if layer is empty (likely absent from the VCF)
        if vcf.layer(vcf_field).any() == '':
            log.warning(f'{vcf_field} is empty, skipping')
            continue
        assay.add_layer(field, vcf.layer(vcf_field, remove_missing=True)[:, :, 0].T)
        log.info(f"Added custom field -- {field}")

    log.info("Adding cell attributes")
    assay.add_row_attr("barcode", vcf.layer(VCFFile.BARCODE, sort=False))

    log.info("Adding variant attributes")
    for name, series in vcf.create_variant_data().iteritems():
        assay.add_col_attr(name, series.values)

    log.info("Adding metadata")
    for name, value in metadata.items():
        assay.add_metadata(name, value)

    sample = metadata.get(SAMPLE, DEFAULT_SAMPLE)
    assay.add_row_attr(SAMPLE, np.array([sample] * assay.shape[0]))

    assay.add_metadata("high_quality_variants", vcf.n_high_quality_variants)

    return assay


def compute_af(ad: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """Compute allele frequency

    Args:
        ad: allele depth
        dp: read depth

    Returns:
        allele frequency
    """
    # suppress true_divide warning
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(ad / dp * 100)


class VCFFile:
    """Class representing vcf file"""

    AD = "calldata/AD"
    DP = "calldata/DP"
    GQ = "calldata/GQ"
    GT = "calldata/GT"
    RGQ = "calldata/RGQ"

    ALT = "variants/ALT"
    CHROM = "variants/CHROM"
    POS = "variants/POS"
    QUAL = "variants/QUAL"
    REF = "variants/REF"

    BARCODE = "samples"

    LAYERS = [AD, DP, GQ, GT, RGQ]
    ANNOTATIONS = [ALT, CHROM, POS, QUAL, REF, BARCODE]

    def __init__(self, filename: str, quality_threshold: Optional[int] = None, custom_fields: Optional[list] = []):
        """Representation for VCF file

        Args:
            filename: path to the vcf file
            quality_threshold: when set, variants above this thresholds come first
        """
        # for now, custom fields can only be calldata (INFO) fields in the VCF
        custom_fields = [f'calldata/{field}' for field in custom_fields]
        fields = list(set(VCFFile.LAYERS + VCFFile.ANNOTATIONS + custom_fields))
        self.__file = allel.read_vcf(filename, fields=fields, fills = {'calldata/TLOD': -100}) # added option to add custom fields
        self.sorting = None
        self.n_high_quality_variants = 0
        self.set_quality_threshold(quality_threshold)

    def set_quality_threshold(self, quality_threshold: Optional[int]):
        """Set quality threshold for sorting the data

        Variants with quality above this threshold will be put to the top.

        Args:
            quality_threshold: when set, variants above this thresholds come first
        """
        if quality_threshold is not None:
            q = self.__file[self.QUAL]

            low_quality = q < quality_threshold
            self.n_high_quality_variants = len(q) - low_quality.sum()
            # q < threshold returns 0 for high quality and 1 for
            # low quality variants, so high quality variants are sorted
            # before low quality ones
            self.sorting = np.argsort(low_quality, kind="mergesort")
        else:
            self.sorting = slice(None)

    def layer(
        self, layer_name: str, *, sort: bool = True, remove_missing: bool = False
    ) -> np.ndarray:
        """Return layer data from vcf file

        Args:
            layer_name: name of the layer to return
            sort: when True, layer is sorted using self.sorting
            remove_missing: when True, missing values (-1) are replaced with 0

        Returns:
            array with layer data
        """
        data = self.__file[layer_name]
        if sort:
            data = data[self.sorting]
        if remove_missing:
            data[data == -1] = 0
        return data

    def create_ngt(self, dtype=np.int8) -> np.ndarray:
        """Create ngt layer from GT data

        Args:
            dtype: dtype of the resulting array

        Returns:
            array with ngt data
        """
        genotype = allel.GenotypeArray(self.layer(self.GT))
        ref, alt = genotype[:, :, 0], genotype[:, :, 1]

        ngt = np.ones(genotype.shape[:2], dtype=dtype)
        ngt[ref == alt] = 2
        ngt[(ref == 0) & (alt == 0)] = 0
        ngt[(ref == -1) & (alt == 0)] = 0
        ngt[(ref == 0) & (alt == -1)] = 0
        ngt[(ref == -1) & (alt == -1)] = 3

        return ngt

    def create_variant_data(self) -> pd.DataFrame:
        """Create pd.DataFrame with variant metadata

        DataFrame contains values for identifying the variant
        (chrom, pos, ref, alt)
        and a unique variant id (chr:pos:ref/alt)

        Returns:
            DataFrame with variant data
        """
        data = pd.DataFrame()
        chrom = Series(self.layer(VCFFile.CHROM))
        data[CHROM] = chrom.map(lambda x: x.lstrip("chrCHR"))
        data[POS] = self.layer(VCFFile.POS)
        data[REF] = self.layer(VCFFile.REF)
        data[ALT] = self.layer(VCFFile.ALT)[:, 0]
        data[QUAL] = self.layer(VCFFile.QUAL)

        def format(row):
            return "chr{0}:{1}:{2}/{3}".format(row[CHROM], row[POS], row[REF], row[ALT])

        data[ID] = data.apply(format, axis=1)
        return data
