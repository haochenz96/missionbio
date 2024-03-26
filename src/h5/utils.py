import logging

log = logging.getLogger(__name__)


def is_sorted(lst):
    """Check if the list is sorted

    Args:
        lst: list to check

    Returns:
        bool: True if list is sorted
    """
    return all(lst[i - 1] <= lst[i] for i in range(1, len(lst)))


def find_common_keys(dictionaries, *, warning=None, names=None):
    """Find a set of keys common to all dictionaries

    Args:
        dictionaries (Iterable[Dict[str, Any]]): dictionaries to take the keys from
        warning (Optional[str]): when set, warning will be logged
        names (Optional[List[str]]): list with a name for each dict. Used in warning message

    Returns:
        Set[str]: common keys
    """
    # convert dictionaries to a list so we can iterate over it twice
    dictionaries = list(dictionaries)
    all_keys = set()
    for d in dictionaries:
        all_keys.update(d)

    common_keys = set(all_keys)
    for i, d in enumerate(dictionaries):
        keys = set(d)
        if keys != common_keys:
            common_keys &= keys
            if warning:
                missing_keys = all_keys - keys
                name = names[i] if names else ""
                log.warning(warning.format(missing_keys=missing_keys, name=name))
    return common_keys
