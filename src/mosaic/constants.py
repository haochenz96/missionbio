"""
Colors and keys for layers and attributes

.. autosummary::
   :toctree:

   COLORS
   PCA_LABEL
   UMAP_LABEL
   SCALED_LABEL
   LABEL
   READS
   PALETTE
   NORMALIZED_READS
   PROTEIN
   DNA
   CNV
"""

PCA_LABEL = 'pca'
UMAP_LABEL = 'umap'
SCALED_LABEL = 'scaled_counts'
LABEL = 'label'
READS = 'read_counts'
PALETTE = 'palette'
NORMALIZED_READS = 'normalized_counts'
AF_MISSING = 'AF_MISSING'
NGT_FILTERED = 'NGT_FILTERED'
GENE_NAME = 'gene_name'
ORG_BARCODE = 'original_barcode'
PLOIDY = 'ploidy'

try:
    import seaborn as sns
    from matplotlib.colors import to_hex

    _colors = sns.color_palette('tab20')
    _col = _colors[::2]
    _col.extend(_colors[1::2])
    _col[2], _col[3] = _col[3], _col[2]
    _col[7], _col[9] = _col[9], _col[7]
    _col[12], _col[13] = _col[13], _col[12]
    _col[17], _col[19] = _col[19], _col[17]

    COLORS = _col.copy()

    _col.append([0, 0, 0])
    _col.extend([sns.desaturate(c, 0.5) for c in COLORS])
    _col.extend([sns.desaturate(c, 0.25) for c in COLORS])

    COLORS = [to_hex(c) for c in _col]
except ImportError:
    COLORS = ["#000000", "#000000", "#000000"]
