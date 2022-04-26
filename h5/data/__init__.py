"""
Classes for representation, reading and writing of MissionBio hdf5  files

Format Specification
====================
Data produced by MissionBio pipelines is stored in a structured
[hdf5](https://www.hdfgroup.org/solutions/hdf5/) file.
This file can contain data for one or more samples. Each sample,
in turn, can contain data for one or more "assays", where each
assay contains data for a different analyte.

Structure
---------
HDF5 files consist of 2 types of objects;
Datasets, which are multidimensional arrays of a single data type
Groups, which are containers that can contain other groups or datasets

A high level overview of the Mission Bio multiomics file format is shown below

```
<root>
+-- metadata
+-- assays
    +-- dna_variants
    +-- dna_read_counts
    +-- protein_read_counts
+-- all_barcodes
    +-- dna_read_counts
    +-- protein_read_counts
```

At the root level of the file, there are 3 hdf5 groups:

-  **metadata** group: contains file metadata below:
    - date_created,
    - file_format_version

- **assays** group stores assay data. When multiple analytes are
merged, assay data is filtered to contain only the cells present in all
analytes. Each analyte's data is stored under a separate sub-group.

- **all_barcodes** stores raw read counts for all cell barcodes. This data is never
filtered, it can be used to compare cell barcode overlap between assays. Each
analyte's data is stored under a separate group.

The **all_barcodes** and **assays** groups use the following structure to store data:

```
+-- metadata
+-- layers
    +--
    +--
+-- ra
    +--
    +--
+-- ca
    +--
    +--
```

- **metadata** group contains assay metadata. For dna this includes genome,
number of total reads, ... For other assays, it will contain different
keys and values.

- **layers** group contains matrices with measurements. Cells are stored in
rows and features (variants, antibodies, amplicons, ...) are stored in
columns. If there are more than one layers, all layers must have the
same shape (same number of rows and columns).

- **ra** group contains row (cell) annotations. It should always include
a unique identifier (usually cell barcode). When a file contains multiple
assays, their row identifiers need to match.

- **ca** group contains column (feature) annotations. If should always include
a unique identifier (called id), but can contain other data, depending
on the assay type.
"""
# This package imports all public members from all submodules.
# Flake complains since they are not used directly.
# flake8: noqa

from .assay import Assay
from .reader import H5Reader
from .writer import H5Writer
