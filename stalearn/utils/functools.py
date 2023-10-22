import numpy as np

def create_function_from_string(code: str):
    import textwrap
    code = textwrap.dedent(code)
    local_ns = {}
    exec(code, globals(), local_ns)
    return local_ns['func']