from importlib import import_module
from pkgutil import walk_packages

from missionbio.cli import tapestri_command


def main():
    """Discover subcommands and run click code.

    Imports cli tools from missionbio.* to allow the
    @click.groups to register themselves
    """
    import missionbio

    # Imports cli tools from missionbio.*.cli to allow the
    # @click.groups to register themselves
    for loader, module_name, is_pkg in walk_packages(missionbio.__path__,
                                                     prefix="missionbio."):
        if module_name.endswith(".cli"):
            import_module(module_name)

    tapestri_command()


if __name__ == "__main__":
    main()
