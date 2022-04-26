"""
Function for creating assays from file(s) produced by the corresponding pipeline.

The same functionality can be accessed from command line using command:
```
tapestri h5 create [dna/protein/rna]
```

Examples
--------

Create a dna assay from dna.vcf data and metadata from metadata.json
```
from missionbio.h5.create import create_dna_assay
assay = create_dna_assay("dna.vcf", metadata="metadata.json")
```
 or
```
tapestri h5 create dna \
    --vcf "dna.vcf" \
    --read-counts "counts.csv" \
    --metadata "metadata.json" \
    --output "dna.hdf5"
```
"""
# This package imports all public members from all submodules.
# Flake complains since they are not used directly.
# flake8: noqa

from .cnv import create_cnv_assay
from .dna import create_dna_assay
from .protein import create_protein_assay
from .rna import create_rna_assay
