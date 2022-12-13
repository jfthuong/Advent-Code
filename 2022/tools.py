from __future__ import annotations

from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generic, List, Literal, Optional, Tuple, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
from IPython import get_ipython  # type: ignore
from IPython.core.interactiveshell import InteractiveShell  # type: ignore
from IPython.core.magic import register_line_cell_magic  # type: ignore

T = TypeVar("T")

# A line magic that can help to download the data and store it in variables
# `input_path` and `data` to use directly in the code after:

# It will use:
# * if no argument: current day and year
# * one argument: the day set as argument + current year
# * two arguments: the day and the year

# ```python
# %get_daily_input 2
# data[:100]
# ```

IMPORT_BLOCK = """\
from aocd import get_data
from aocd import submit
from IPython.display import HTML
"""


@register_line_cell_magic
def get_daily_input(line: str) -> None:
    """Retrieve the input for the current day and storing variables
    * input_path: Path to the input file
    * data: Content of the input file
    """
    ipython = cast(InteractiveShell, get_ipython())

    ipython.run_cell(IMPORT_BLOCK)

    today = datetime.today()
    current_year = today.year
    current_day = today.day
    if line:
        args = [int(e) for e in line.split()] + [current_year]
    else:
        args = [current_day, current_year]
    day, year = args[:2]

    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True)
    input_path = input_dir / f"{day:02d}.txt"
    input_path_html = f'<a href="{input_path.as_posix()}">{input_path.as_posix()}</a>'

    if not input_path.exists():
        ipython.system(f"aocd {day} {year} > {input_path}")
        ipython.run_cell(f"HTML('✅Input saved to {input_path_html}')")
    else:
        ipython.run_cell(f"HTML('✔️Input already exists in {input_path_html}')")

    ipython.run_cell(f"input_path = Path('{input_path.as_posix()}')")
    ipython.run_cell(f"data = get_data(day={day}, year={year})")
    msg = "✅Variables <code>input_path</code> and <code>data</code> set"
    ipython.run_cell(f"HTML('{msg}')")


@overload
def split_data(
    data: str,
    *,
    by_pairs: Literal[True],
    safe_eval: Literal[True],
) -> List[Tuple[Any, Any]]:
    ...


@overload
def split_data(
    data: str,
    *,
    safe_eval: Literal[True],
) -> List[Any]:
    ...


@overload
def split_data(
    data: str,
    *,
    by_pairs: Literal[True],
    safe_eval: Literal[False],
) -> List[Tuple[str, str]]:
    ...


@overload
def split_data(data: str, *, safe_eval: Literal[False]) -> List[str]:
    ...


@overload
def split_data(data: str) -> List[str]:
    ...


def split_data(
    data: str,
    separator: str = "\n",
    strip: bool = True,
    skip_empty: bool = True,
    by_pairs: bool = False,
    safe_eval: bool = False,
) -> List[Any]:
    """Split the data into a list of lines or pairs of lines"""
    if strip:
        data = data.strip()

    if by_pairs and separator == "\n":
        separator = "\n\n"

    blocks = data.split(separator)

    if skip_empty:
        blocks = [b for b in blocks if b.strip()]

    if by_pairs:
        return [
            tuple(literal_eval(e) if safe_eval else e for e in b.splitlines())
            for b in blocks
        ]

    return [literal_eval(b) if safe_eval else b for b in blocks]


class Grid(Generic[T]):
    """Class for manipulation of Grids, stored as numpy arrays"""

    def __init__(self, data: str) -> None:
        self._data: npt.NDArray[np.str_] = np.array([list(r) for r in split_data(data)], dtype=np.str_)

    def __repr__(self) -> str:
        return str(self._data)