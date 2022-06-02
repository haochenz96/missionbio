from importlib import import_module
from pkgutil import walk_packages

from . import tapestri_command


def main():
    """Discover subcommands and run click code.

    Imports cli tools from src.* to allow the
    @click.groups to register themselves
    """
    import h5.cli

    # Imports cli tools from src.*.cli to allow the
    # @click.groups to register themselves
    # for loader, module_name, is_pkg in walk_packages(src.__path__,
    #                                                  prefix="src."):
    #     if module_name.endswith(".cli"):
    #         import_module(module_name)

    tapestri_command()


if __name__ == "__main__":
    main()
