"""
Python package `missionbio.h5` provides objects, functions and a command-line
interface for working with missionbio hdf5 files.

Command-line interface
----------------------
The library can also be used from the command line. You can access most
of the functionality via the `tapestri h5` command. For instance, to create
dna assay from vcf file, you can run:

```
tapestri h5 create dna \\
    --vcf <data.vcf> \\
    --read-counts <readcounts.csv> \\
    --metadata <metadata.json> \\
    --output <data.h5>
```
"""
__version__ = "3.2.0"
