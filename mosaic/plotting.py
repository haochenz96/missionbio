import functools
from typing import Callable, Optional

try:
    import seaborn as sns
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    import_error = None
except ImportError as err:
    sns = None
    plt = None
    mpimg = None
    to_hex = None
    import_error = err


def require_seaborn(f: Optional[Callable] = None):
    """
    The seaborn library is required for plotting.

    Parameters
    ----------
    f : function
        function to decorate

    Raises
    ------
    ValueError
        Raise exception if seaborn could not be imported.

    Also
    ----
    Can be used as a decorator.
    """
    def check_seaborn_imported():
        if sns is None:
            raise ValueError("The seaborn library is required for plotting.") from import_error

    if f is None:
        check_seaborn_imported()
    else:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            check_seaborn_imported()

            return f(*args, **kwargs)

        return wrapper
