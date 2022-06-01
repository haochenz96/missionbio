import click


@click.group(context_settings={"help_option_names": ("-h", "--help")})
def tapestri_command():
    pass
