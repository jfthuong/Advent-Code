from __future__ import annotations

import json
import re
from ast import literal_eval
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Self,
    TypeVar,
    Union,
    cast,
    overload,
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore
import pandas as pd
from IPython import get_ipython  # type: ignore
from IPython.core.display import display  # type: ignore
from IPython.core.interactiveshell import InteractiveShell  # type: ignore
from IPython.core.magic import register_line_cell_magic  # type: ignore
from IPython.core.magic import register_line_magic
from matplotlib.colors import ListedColormap
from PIL import Image

T = TypeVar("T")
Y = TypeVar("Y")

Palette = list[str]

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


def pairwise(data: list[T], nb_elements: int = 2) -> list[tuple[T, ...]]:
    """Split the data into pairs of elements"""
    return [tuple(data[i : i + nb_elements]) for i in range(0, len(data), nb_elements)]


@overload
def split_data(
    data: str,
    *,
    by_tuple: Literal[True],
    safe_eval: Literal[True],
) -> list[tuple[Any, ...]]: ...


@overload
def split_data(
    data: str,
    *,
    safe_eval: Literal[True],
) -> list[Any]: ...


@overload
def split_data(
    data: str,
    *,
    by_tuple: Literal[True],
    safe_eval: Literal[False],
) -> list[tuple[str, ...]]: ...


@overload
def split_data(data: str, *, safe_eval: Literal[False]) -> list[str]: ...


@overload
def split_data(data: str) -> list[str]: ...


def split_data(
    data: str,
    separator: Optional[str] = None,
    strip: bool = True,
    skip_empty: bool = True,
    tuple_separator: str = "\n",
    by_tuple: bool = False,
    safe_eval: bool = False,
) -> list[Any]:
    """Split the data into a list of lines or pairs of lines

    Args:
        data: The data from challenge to be split
        separator: separator between each data (default: \\n for line return)
        strip: whether to strip the spaces around initial data (default: True)
        skip_empty: skip empty lines (default: True)
        tuple_separator: separator to split the blocks (default: \\n\\n)
        by_tuple: cut by blocks (by default using separator of an empty line)
            and return a tuple for element split by tuple_separator (default: False)
        safe_eval: try to interpolate the data (integer, list, etc.) (default: False)
    """
    if strip:
        data = data.strip()

    if separator is None:
        separator = "\n\n" if by_tuple else "\n"

    blocks = data.split(separator)

    if skip_empty:
        blocks = [b for b in blocks if b.strip()]

    if by_tuple:
        return [
            tuple(literal_eval(e) if safe_eval else e for e in b.split(tuple_separator))
            for b in blocks
        ]

    return [literal_eval(b) if safe_eval else b for b in blocks]


def split_data_tuple_lists(
    data: str,
    separator: str = "\n",
    tuple_separator: str = "\t",
    strip: bool = True,
    skip_empty: bool = True,
    safe_eval: bool = False,
) -> tuple[list[Any], ...]:
    """Split the data into a tuple of list of data

    Args:
        data: The data from challenge to be split
        separator: separator between each data (default: \\n for line return)
        tuple_separator: separator to split the blocks (default: \\t)
        strip: whether to strip the spaces around initial data (default: True)
        skip_empty: skip empty lines (default: True)
        safe_eval: try to interpolate the data (integer, list, etc.) (default: False)
    """
    if strip:
        data = data.strip()

    blocks = iter(data.split(separator))

    if skip_empty:
        blocks = (b for b in blocks if b.strip())

    split_blocks = [
        [literal_eval(e) if safe_eval else e for e in re.split(tuple_separator, b)]
        for b in blocks
    ]
    return tuple(list(d) for d in zip(*split_blocks))


def split_data_2_df(data: str, sep: str = r"\s+", as_int: bool = False) -> pd.DataFrame:
    """Split the data into a DataFrame

    Args:
        data: The data from challenge to be split
        sep: regex separator between each data (default: \\s+)
        is_int: whether to convert the data to integer (default: False)
    """
    df = pd.DataFrame([re.split(sep, b) for b in split_data(data)])
    if as_int:
        df = df.astype(int)
    return df


DFLT_PALETTE_NAME = "rocket"


def get_palette(
    nb_colors: int = 1000, name_palette=DFLT_PALETTE_NAME, reverse: bool = False
) -> Palette:
    """Get a list of colors to use for plotting"""
    # Note: palette obtained from SeaBorn
    # https://seaborn.pydata.org/tutorial/color_palettes.html
    # import seaborn as sns
    # palette = sns.color_palette(name_palette, 1000).as_hex()
    # Path(f"palette_{name_palette}.json").write_text(json.dumps(palette))
    path_palette = Path(f"palette_{name_palette}.json")
    if not path_palette.exists():
        raise ValueError(f"Palette {name_palette} not found")

    palette = json.loads(path_palette.read_text())
    if reverse:
        palette = list(reversed(palette))

    return [palette[i * len(palette) // nb_colors] for i in range(nb_colors)]


@register_line_magic
def show_palette(line: str) -> None:
    """Show a palette"""
    ipython = cast(InteractiveShell, get_ipython())
    palette = ipython.ev(line)
    if not isinstance(palette, list):
        raise ValueError("Palette must be a list of colors")
    palette = cast(Palette, palette)

    mode = "RGBA" if any(c.startswith("#") and len(c) == 9 for c in palette) else "RGB"

    width_px = 1000
    len_palette = len(palette)
    img = Image.new(mode=mode, size=(width_px, 120))

    for i, color in enumerate(palette):
        newt = Image.new(mode=mode, size=(width_px // len_palette, 100), color=color)
        img.paste(newt, (i * width_px // len_palette, 10))

    display(img)


def get_color_from_pct(
    red: float = 0, green: float = 0, blue: float = 0, alpha: Optional[float] = None
) -> str:
    """Get a color from a percentage of red, green, blue, and potentially alpha"""
    colors = [red, green, blue] if alpha is None else [red, green, blue, alpha]
    return "#" + "".join(f"{int(c * 255):02X}" for c in colors)


DFLT_OVERLAY = get_color_from_pct(blue=1, alpha=0.2)

Pos = tuple[int, int]
GridAny = npt.NDArray[Any]
GridFunc = Callable[[T, int, int], Y]
GridFuncAny = GridFunc[T, Any]
GridFuncOpt = Optional[GridFunc[T, Any]]
SelectStartEnd = Union[str, int, float, GridFunc[T, bool]]


class Point(NamedTuple):
    """Representaiton of a 2D point with integer coordinates"""

    x: int
    y: int

    def __add__(self, other: Point) -> Point:  # type: ignore
        """Translate the point with `Point + Point`"""
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        """Translate the point with `Point - Point`"""
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int) -> Point:  # type: ignore
        """Multiply dimensions by using `Point * int`"""
        return Point(self.x * other, self.y * other)

    def __rmul__(self, other: int) -> Point:  # type: ignore
        """Multiply dimensions by using `int * Point`"""
        return self * other

    def __truediv__(self, other: float) -> Point:
        """Forbidden operation of `Point / float`"""
        raise TypeError("Cannot divide a point by a float")

    def __floordiv__(self, other: int) -> Point:
        """Divide dimensions by using `Point // int`"""
        return Point(self.x // other, self.y // other)

    def __neg__(self) -> Point:
        """Get the opposite of the point with `-Point`"""
        return Point(-self.x, -self.y)

    def neighbors(self, diagonal: bool = False) -> list[Point]:
        """Get the list of surrounding positions"""
        positions: list[Point] = []
        for dx, dy in product([-1, 0, 1], repeat=2):
            if dx == dy == 0:
                continue
            if not diagonal and dx * dy != 0:
                continue
            positions.append(self + Point(dx, dy))
        return positions


class Point3D(NamedTuple):
    """Representation of a 3D point with float coordinates"""

    x: float
    y: float
    z: float

    def __add__(self, other: Point3D) -> Point3D:  # type: ignore
        """Translate the point with `Point3D + Point3D`"""
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        """Translate the point with `Point3D - Point3D`"""
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> Point3D:  # type: ignore
        """Multiply dimensions by using `Point3D * float`"""
        return Point3D(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float) -> Point3D:  # type: ignore
        """Multiply dimensions by using `float * Point3D`"""
        return self * other

    def __truediv__(self, other: float) -> Point3D:
        """Divide dimensions by using `Point3D / float`"""
        return Point3D(self.x / other, self.y / other, self.z / other)

    def __floordiv__(self, other: int) -> Point3D:
        """Divide dimensions by using `Point3D // int`"""
        return Point3D(self.x // other, self.y // other, self.z // other)

    def __neg__(self) -> Point3D:
        """Get the opposite of the point with `-Point3D`"""
        return Point3D(-self.x, -self.y, -self.z)


class Grid(Generic[T]):
    """Class for manipulation of Grids, stored as numpy arrays

    Args:
        data: The data from challenge to be split
        func: Function to apply to each element of the array with 3 arguments:
            * the element value
            * the row index
            * the column index
    """

    def __init__(
        self, data: Union[str, npt.NDArray[np.str_]], func: GridFuncOpt[T] = None
    ) -> None:
        if isinstance(data, str):
            self.data: npt.NDArray[np.str_] = np.array(
                [list(r) for r in split_data(data)], dtype=np.str_
            )
        else:
            self.data = data
        self.encoded: GridAny = self.apply(func) if func else self.data

    def __repr__(self) -> str:
        return str(self.data)

    def __getitem__(self, pos: Union[Pos, Point]) -> T:
        """Access a given point with `grid[(x, y)]`"""
        return self.data[pos]

    def __setitem__(self, pos: Union[Pos, Point], value: T) -> None:
        """Set a given point with `grid[(x, y)] = value`"""
        self.data[pos] = value

    def __contains__(self, point: Union[Pos, Point]) -> bool:
        """Check a point is inside the grid with `Point(x, y) in grid`"""
        point = Point(*point)
        return 0 <= point.x < self.data.shape[0] and 0 <= point.y < self.data.shape[1]

    def copy(self) -> Grid[T]:
        """Get a copy of the grid"""
        copied_grid: Grid[T] = Grid(self.data.copy())
        copied_grid.encoded = self.encoded.copy()
        return copied_grid

    @classmethod
    def from_char(cls, character: str, width: int, height: int) -> "Grid":
        """Create a grid with a single character"""
        data = np.full((width, height), character)
        return cls(data, func=None)

    def apply(self, func: GridFunc[T, Any]) -> GridAny:
        """Apply a function to each element of the array and the coordinates

        Args:
            func: Function to apply to each element of the array (3 arguments)

        Returns:
            The array with the function applied to each element
        """
        return np.vectorize(func)(self.data, *np.indices(self.data.shape))

    def show_with_colors(
        self,
        palette_func: GridFunc[T, int],
        palette_name: str = DFLT_PALETTE_NAME,
        palette_reverse: bool = False,
        palette_additional_colors: Optional[Palette] = None,
        with_original: bool = False,
        overlay_func: Optional[GridFunc[T, bool]] = None,
        overlay_color: str = DFLT_OVERLAY,
        overlay_text: bool = False,
    ) -> None:
        """Show the grid with colors

        Args:
            palette_func: Function to get the color index of elements (3 arguments)
            palette_name: Name of the palette to use (default: "rocket")
            palette_reverse: Whether to reverse the palette (default: False)
            palette_additional_colors: Additional colors to use for the palette (optional)
            overlay_func: Function to get a bool for overlay of elements (3 arguments)
                Ideally, the color should be transparent.
            overlay_color: Color to use for overlay
            overlay_text: Whether to show the text when overlaying (default: False)
            with_original: Whether to use the data from original grid
                False to show the encoded data (default)
                True to show the original data
        """
        data = self.data if with_original else self.encoded
        palette = get_palette(
            len(np.unique(data)), name_palette=palette_name, reverse=palette_reverse
        )
        if palette_additional_colors:
            palette += palette_additional_colors

        data_with_index = np.vectorize(palette_func)(data, *np.indices(data.shape))
        palette_plt = ListedColormap(palette)  # type: ignore

        plt.figure(figsize=data.shape[::-1])
        plt.imshow(data_with_index, aspect="auto", cmap=palette_plt)

        if overlay_func:
            overlay = np.zeros(data_with_index.shape, dtype=int)
            has_overlay = np.vectorize(overlay_func)(data, *np.indices(data.shape))
            if overlay_text:
                for i, j in zip(*np.where(has_overlay)):  # type: ignore
                    plt.text(j, i, data[i, j], color="white", ha="center", va="center")
            overlay[has_overlay] = 1
            plt.imshow(
                overlay,
                aspect="auto",
                cmap=ListedColormap(["#00000000", overlay_color]),  # type: ignore
            )

    def find(self, pattern: str) -> list[Point]:
        """find all the points matching a given pattern"""
        width, height = self.data.shape
        return [
            Point(i, j)
            for i, j in product(range(width), range(height))
            if self.data[i, j] == pattern
        ]

    def find_pattern(
        self,
        pattern: str,
        diagonal: bool = True,
        horizontal: bool = True,
        vertical: bool = True,
    ) -> list[list[Point]]:
        """Find all the patterns in the grid, searching in all directions"""
        width, height = self.data.shape
        found = []

        directions: list[tuple[int, int]] = []
        if diagonal:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        if horizontal:
            directions.extend([(1, 0), (-1, 0)])
        if vertical:
            directions.extend([(0, 1), (0, -1)])

        for i, j in product(range(width), range(height)):
            for dx, dy in directions:
                found.append(self.find_pattern_direction(pattern, i, j, dx, dy))

        return [f for f in found if f]

    def find_pattern_direction(
        self, pattern: str, x: int, y: int, dx: int, dy: int
    ) -> list[Point]:
        """Find a pattern in the grid in a given direction"""
        pattern_len = len(pattern)
        width, height = self.data.shape
        found = []

        for i in range(pattern_len):
            xi, yi = x + i * dx, y + i * dy
            if not (0 <= xi < width and 0 <= yi < height):
                return []
            if self.data[xi, yi] != pattern[i]:
                return []

            found.append(Point(xi, yi))

        return found


class GridWithNetwork(Grid):
    """Class for manipulation of Grids, stored as numpy arrays, with a graph

    Args:
        data: The data from challenge to be split
        func: Function to apply to each element of the array with 3 arguments:
            * the element value
            * the row index
            * the column index
        directed: Whether the graph is directed (default: True)
        start: Function to get the start of the edge; could be:
            * a string, int, or float to filter based on value (default: "S")
            * a list of positions to use directly
            * a function to filter based on value, row, and column (3 arguments)
        end: Function to get the end of the edge (same as select_start, default: "E")
        with_original: Whether to use the data from original grid to get start and end
            (default: False, i.e. use the encoded data)
    """

    def __init__(
        self,
        data: str,
        func: GridFuncOpt[T] = None,
        directed: bool = True,
        start: SelectStartEnd = "S",
        end: SelectStartEnd = "E",
        with_original: bool = False,
    ) -> None:
        super().__init__(data, func)
        self.graph = nx.DiGraph() if directed else nx.Graph()

        datagrid = self.data if with_original else self.encoded

        def get_start_end(select: SelectStartEnd) -> Union[Pos, list[Pos]]:
            if isinstance(select, (str, int, float)):
                mask = datagrid == select
            else:
                mask = np.vectorize(select)(datagrid, *np.indices(datagrid.shape))

            candidates = np.argwhere(mask).tolist()
            if len(candidates) == 0:
                print("--DATA--")
                print(datagrid)
                print("--MASK--")
                print(mask)
                raise ValueError(f"Could not find start/end with {select!r}")

            if len(candidates) == 1:
                return cast(Pos, tuple(candidates[0]))
            return candidates

        self.start = get_start_end(start)
        self.end = get_start_end(end)
        self.start_end_with_original = with_original

    @property
    def nodes(self) -> nx.NodeView:
        """Get the nodes of the graph"""
        return self.graph.nodes  # type: ignore

    def successors(self, *args, **kwargs):
        """Get the successors of the graph"""
        assert isinstance(self.graph, nx.DiGraph), "Graph is not directed"
        return self.graph.successors(*args, **kwargs)

    def get_neighbors(self, pos: Pos) -> list[Pos]:
        """Get neighbors of a position"""
        width, height = self.data.shape
        x, y = pos

        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbors if 0 <= n[0] < width and 0 <= n[1] < height]

    def build_graph(
        self,
        edge_func: Callable[[T, int, int, T, int, int], bool],
        node_color_func: Optional[GridFuncAny] = None,
        with_original: bool = False,
    ) -> "GridWithNetwork":
        """Add a graph
        Args:
            edge_func: Function to determine if an edge shall be added, with 6 arguments:
                * the value of start
                * the row index of start
                * the column index of start
                * the value of end
                * the row index of end
                * the column index of end
            node_color_func: Function to get the color of the edge (3 arguments)
                If not provided, the color will be black
            with_original: Whether to use the data from original grid (default: False)
        """
        data = self.data if with_original else self.encoded

        for pos, val in np.ndenumerate(data):
            pos = cast(Pos, pos)
            i, j = pos
            color = node_color_func(val, i, j) if node_color_func else "black"
            label = self.data[pos]  # we use the original name
            # print(f"Adding node {pos} with label {label} and color {color}")
            self.graph.add_node(pos, label=label, color=color)
            for neighbor in self.get_neighbors(pos):
                if edge_func(val, i, j, data[neighbor], *neighbor):
                    self.graph.add_edge(pos, neighbor)

        return self

    def show_graph(self):
        """Display the graph"""
        is_small = self.data.shape[0] < 10
        size = 10 if is_small else 20

        plt.figure(figsize=(size, size))
        nx.draw(
            self.graph,
            pos=nx.spring_layout(self.graph),
            node_color=[self.graph.nodes[n]["color"] for n in self.graph.nodes],
            with_labels=is_small,
        )

    def show_grid_with_colors(
        self,
        palette_func: GridFunc[T, int],
        palette_name: str = DFLT_PALETTE_NAME,
        palette_reverse: bool = False,
        palette_additional_colors: Optional[Palette] = None,
        with_original: bool = False,
        shortest_path_color: Optional[str] = DFLT_OVERLAY,
        overlay_text: bool = True,
    ) -> None:
        """Show the grid with colors

        Args:
            palette_func: Function to get the color index of elements (3 arguments)
            palette_name: Name of the palette to use (default: "rocket")
            palette_reverse: Whether to reverse the palette (default: False)
            palette_additional_colors: Additional colors to use for the palette (optional)
            with_original: Whether to use the data from original grid
                False to show the encoded data (default)
                True to show the original data
            shortest_path_color: Color to use for overlay
                (set to None to not show shortest path)
            overlay_text: Whether to show the text when overlaying shortest path
                (default: True)
        """
        if shortest_path_color is None:
            overlay_func: Optional[GridFuncAny] = None
            overlay_color = DFLT_OVERLAY
        else:
            shortest_path = self.shortest_path
            overlay_func = lambda _, i, j: (i, j) in shortest_path
            overlay_color = shortest_path_color

        self.show_with_colors(
            palette_func=palette_func,
            palette_name=palette_name,
            palette_reverse=palette_reverse,
            palette_additional_colors=palette_additional_colors,
            with_original=with_original,
            overlay_func=overlay_func,
            overlay_text=overlay_text,
            overlay_color=overlay_color,
        )

    @property
    def shortest_path(self) -> list[Pos]:
        """Get the shortest path from start to end.
        Note that there are possibly multiple start or end points.
        """
        shortest_path: list[Pos] = []

        if not isinstance(self.start, list):
            self.start = [self.start]
        if not isinstance(self.end, list):
            self.end = [self.end]

        for start, end in product(self.start, self.end):
            try:
                path = cast(list[Pos], nx.shortest_path(self.graph, start, end))
            except nx.NetworkXNoPath:
                continue
            if len(shortest_path) == 0 or len(path) < len(shortest_path):
                shortest_path = path

        return shortest_path

    @property
    def shortest_path_length(self) -> int:
        """Get the length of the shortest path from start to end.
        Note that there are possibly multiple start or end points.
        """
        if not isinstance(self.start, list):
            self.start = [self.start]
        if not isinstance(self.end, list):
            self.end = [self.end]

        shortest_path_length = 0
        for start, end in product(self.start, self.end):
            try:
                path_length = cast(int, nx.shortest_path_length(self.graph, start, end))
            except nx.NetworkXNoPath:
                continue
            if shortest_path_length == 0 or path_length < shortest_path_length:
                shortest_path_length = path_length

        return shortest_path_length
