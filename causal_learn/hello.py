def make_greeting(name: str = None) -> str:
    """

    Parameters
    ----------
    name : str :
         (Default value = None) name to be greeted

    Returns
    -------
    str : string with greeting sentence
    """
    greeting_str = "Hello!"
    if name:
        greeting_str = f"Hello, {name}!"

    return greeting_str
