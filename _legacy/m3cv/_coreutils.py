def simplelog(msg, verbose):
    """Simple version of a logger. At a later time I will update
    this to use Python's native logging capabilities

    Conditionally prints message based on bool value of verbose.
    """
    if verbose:
        print(msg)
