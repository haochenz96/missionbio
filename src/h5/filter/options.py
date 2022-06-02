import functools
from typing import Callable

import click

from h5.filter import DefaultFilter


def variant_filtering_options(function: Callable):
    """Click options for filtering decorator

    Args:
        function: function to add filtering options to

    Returns:
        decorated function
    """

    @click.option(
        "--input",
        "h5_file",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="path to the input h5 file",
    )
    @click.option(
        "--gq",
        "gq_cutoff",
        default=DefaultFilter.gq_cutoff,
        show_default=True,
        type=click.IntRange(min=0, max=100),
        help="Variants with quality less than cutoff are discarded",
    )
    @click.option(
        "--dp",
        "dp_cutoff",
        default=DefaultFilter.dp_cutoff,
        show_default=True,
        type=float,
        help="Variants with read depth less than cutoff are discarded",
    )
    @click.option(
        "--af",
        "af_cutoff",
        default=DefaultFilter.af_cutoff,
        type=click.FloatRange(min=0, max=100),
        show_default=True,
        help="Variants with allele frequency less than cutoff are discarded",
    )
    @click.option(
        "--missing-cells-cutoff",
        "missing_cells_cutoff",
        default=DefaultFilter.missing_cells_cutoff,
        type=click.FloatRange(min=0, max=100),
        show_default=True,
        help="Variants with % cells missing more than cutoff are discarded",
    )
    @click.option(
        "--missing-variants-cutoff",
        "missing_variants_cutoff",
        default=DefaultFilter.missing_variants_cutoff,
        type=click.FloatRange(min=0, max=100),
        show_default=True,
        help="Cells with % variants missing more than cutoff are discarded",
    )
    @click.option(
        "--mutated-cells-cutoff",
        "mutated_cells_cutoff",
        default=DefaultFilter.mutated_cells_cutoff,
        type=click.FloatRange(min=0, max=100),
        show_default=True,
        help="Cells with % mutations less than cutoff are discarded",
    )
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper
