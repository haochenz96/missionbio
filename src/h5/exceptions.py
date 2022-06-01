class UserError(ValueError):
    """
    A known error that should be shown to the user as a message.
    """


class ValidationError(UserError):
    """
    Validation error in assay structure
    """
