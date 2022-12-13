from pathlib import Path
from typing import cast
from datetime import datetime

from IPython import get_ipython  # type: ignore
from IPython.core.interactiveshell import InteractiveShell  # type: ignore
from IPython.core.magic import register_line_cell_magic  # type: ignore


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
def get_daily_input(line:str) -> None:
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