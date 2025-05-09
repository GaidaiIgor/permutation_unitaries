""" General utilities. """
from __future__ import annotations

import ast
import inspect
import textwrap
import traceback
from typing import Sequence

import numpy as np
from scipy import stats


def get_error_margin(data: Sequence, confidence: float = 0.95) -> float:
    z_val = stats.norm.ppf((1 + confidence) / 2)
    return z_val * np.std(data, ddof=1) / len(data) ** 0.5


def array_to_str(arr: Sequence) -> str:
    """ Converts a given sequence to a string. """
    return ''.join([str(val) for val in arr])


def make_dict(*args):
    """ Creates a dictionary out of given arguments, using variable names passed to the call as keys. """
    current_frame = inspect.currentframe()
    func_name = current_frame.f_code.co_name
    caller_frame = current_frame.f_back
    caller_start_line_num = caller_frame.f_code.co_firstlineno
    call_line_num_abs = traceback.extract_stack()[-2].lineno
    source = textwrap.dedent(inspect.getsource(caller_frame))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == func_name and node.lineno + caller_start_line_num - 1 == call_line_num_abs:
            return {arg.id: val for arg, val in zip(node.args, args)}
