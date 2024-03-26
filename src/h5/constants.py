"""Constants holding names of group/keys in data files"""

# predefined h5 groups
ASSAYS = "assays"
LAYERS = "layers"
ROW_ATTRS = "ra"
COL_ATTRS = "ca"
METADATA = "metadata"
RAW_COUNTS = "all_barcodes"

# special row annotation
BARCODE = "barcode"
SAMPLE = "sample_name"

# special column annotation
ID = "id"

# special metadata
DATE_CREATED = "date_created"
SDK_VERSION = "sdk_version"

# dna assay layers
AD = "AD"
AF = "AF"
DP = "DP"
GQ = "GQ"
RGQ = "RGQ"
NGT = "NGT"
RO = "RO"
FILTER_MASK = "FILTER_MASK"
FILTERED_KEY = "filtered"

# dna specific row attributes
CHROM = "CHROM"
POS = "POS"
REF = "REF"
ALT = "ALT"
QUAL = "QUAL"

# Names of the columns in whitelist
START = "start"
END = "end"
ALLELE_INFO = "allele_info"

# variant metadata
VARIANT = "Variant"
GENE = "gene"
CELL = "Cell"
GENOME = "genome"
WHITELIST = "Whitelist"

DNA_ASSAY = "dna_variants"
DNA_READ_COUNTS_ASSAY = "dna_read_counts"
PROTEIN_ASSAY = "protein_read_counts"
RNA_ASSAY = "rna_read_counts"

# sample attributes
N_AMPLICONS = "n_amplicons"
N_PASSING_VARIANTS = "n_passing_variants"
N_PASSING_CELLS = "n_passing_cells"
N_PASSING_VARIANTS_PER_CELL = "n_passing_variants_per_cell"
N_READ_PAIRS = "n_read_pairs"
N_READ_ASSIGNED_TO_CELLS = "n_read_assigned_to_cells"
N_VARIANTS_PER_CELL = "n_variants_per_cell"

# default metadata
DEFAULT_SAMPLE = "sample1"
