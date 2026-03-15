"""
Implementation details for `ReportTable` styling.

This module defines low-level representations of table styles
and metric formatting configurations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Literal, Any
import math
import numbers


@dataclass
class MetricFormat:
    """
    Formatting configuration for a single metric (row or column).

    This object describes how to convert numeric values associated to a given
    metric into human-readable strings (percent vs level, number of decimals,
    custom formatter, etc.). It is used by `ReportTable` when formatting table
    cells.

    Parameters
    ----------

    percent : bool, default False
        Whether values for this metric should be interpreted as proportions
        and rendered as percentages. If True, values are typically multiplied
        by 100 and a '%' suffix is appended.

    decimals : int or None, default 2
        Number of decimal places to display when formatting numeric values.
        If None, a default representation may be used by the implementation.

    formatter : callable or None, default None
        Optional custom formatter of the form `formatter(value) -> str`.
        When provided, this function is responsible for producing the final
        string representation of the value, and `percent`/`decimals` are
        ignored for this metric.
    """

    percent: bool = False
    decimals: int | None = 2
    formatter: Callable[[Any], str] | None = None

    def format(self, value: Any) -> str:
        """
        Format a single value according to this metric format.

        Implementations should handle numeric values as well as pre-formatted
        strings. For strings, most implementations will simply return the
        input unchanged.

        Parameters
        ----------

        value : Any
            Value to format (numeric or already formatted as string).

        Returns
        -------

        str
            Formatted representation of the input value.
        """
        if self.formatter is not None:
            return self.formatter(value)

        if value is None:
            return ""

        if isinstance(value, str):
            return value

        try:
            if isinstance(value, numbers.Real):
                if math.isnan(float(value)):
                    return ""
                v = float(value)
            else:
                return str(value)
        except Exception:
            return str(value)

        if self.percent:
            v *= 100.0

        if self.decimals is None:
            txt = f"{v}"
        else:
            fmt = f".{self.decimals}f"
            txt = format(v, fmt)

        if self.percent:
            txt += "%"

        return txt


@dataclass
class TableStyle:
    """
    Internal representation of a table style.

    A `TableStyle` captures low-level rendering decisions such as borders,
    horizontal and vertical lines, emphasis of header and index, zebra
    striping, default alignment and representation of missing values.

    End-users typically select a style via a string name (e.g. 'academic')
    and do not instantiate this class directly. Style objects are built
    internally from a registry and optionally overridden via `style_options`.

    Parameters
    ----------

    name : str
        Human-readable name of the style (e.g. 'academic', 'minimal').

    outer_border : {'none', 'box', 'top_bottom'}, default 'top_bottom'
        How to render outer borders around the table.

    inner_hlines : {'none', 'all', 'header', 'header_footer'}, default 'header_footer'
        Policy for horizontal rules within the table body.

    inner_vlines : {'none', 'all', 'inner'}, default 'none'
        Policy for vertical rules within the table body.

    bold_header : bool, default True
        Whether column headers should be emphasized (e.g. bold).

    bold_index : bool, default True
        Whether the index (stub column) should be emphasized.

    last_row_emphasis : bool, default False
        Whether the last row should be visually emphasized (e.g. rule above,
        bold text). Useful for "Total" or "Overall" rows for example.

    zebra_striping : bool, default False
        Whether to apply alternating row background colors in the HTML
        representation (LaTeX handling is implementation-dependent).

    header_align : {'left', 'center', 'right'}, default 'center'
        Default alignment for column headers.

    body_align : {'left', 'center', 'right'}, default 'right'
        Default alignment for numeric cells in the table body.

    title_bold : bool, default True
        Whether the table title should be bold.

    title_underline : bool, default False
        Whether the table title should be underlined.

    na_rep : str, default '--'
        String representation used for missing values (NaN, None, etc.).
    """

    name: str
    outer_border: Literal["none", "box", "top_bottom"] = "top_bottom"
    inner_hlines: Literal["none", "all", "header", "header_footer"] = "header_footer"
    inner_vlines: Literal["none", "all", "inner"] = "none"
    bold_header: bool = True
    bold_index: bool = True
    last_row_emphasis: bool = False
    zebra_striping: bool = False
    header_align: Literal["left", "center", "right"] = "center"
    body_align: Literal["left", "center", "right"] = "right"
    title_bold: bool = True
    title_underline: bool = False
    na_rep: str = "--"

    def with_overrides(self, options: Mapping[str, Any] | None) -> TableStyle:
        """
        Return a copy of this style with selected fields overridden.

        This method is used internally by `QuantTable` to apply
        `style_options` provided by the user on top of a base style
        resolved from a style name.

        Parameters
        ----------

        options : mapping or None
            A mapping of field names to new values. Unknown keys are
            typically ignored by implementations.

        Returns
        -------

        TableStyle
            A new style object with overrides applied.
        """
        if not options:
            return self

        valid_fields = set(self.__dataclass_fields__.keys())

        overrides: dict[str, Any] = {
            key: value
            for key, value in options.items()
            if key in valid_fields
        }

        if not overrides:
            return self

        merged = {**self.__dict__, **overrides}
        return TableStyle(**merged)

# ------------------------------------------------------------------
# |                       Predefined themes                        |
# ------------------------------------------------------------------

_STYLE_REGISTRY: dict[str, TableStyle] = {
    "academic": TableStyle(
        name="academic",
        outer_border="top_bottom",
        inner_hlines="header_footer",
        inner_vlines="none",
        bold_header=True,
        bold_index=True,
        last_row_emphasis=False,
        zebra_striping=False,
        header_align="center",
        body_align="right",
        title_bold=True,
        title_underline=False,
        na_rep="--",
    ),
    "minimal": TableStyle(
        name="minimal",
        outer_border="none",
        inner_hlines="none",
        inner_vlines="none",
        bold_header=True,
        bold_index=False,
        last_row_emphasis=False,
        zebra_striping=False,
        header_align="center",
        body_align="right",
        title_bold=False,
        title_underline=False,
        na_rep="--",
    ),
    "boxed": TableStyle(
        name="boxed",
        outer_border="box",
        inner_hlines="all",
        inner_vlines="inner",
        bold_header=True,
        bold_index=True,
        last_row_emphasis=True,
        zebra_striping=False,
        header_align="center",
        body_align="right",
        title_bold=True,
        title_underline=True,
        na_rep="--",
    ),
}

def get_table_style(style_name: str) -> TableStyle:
    """
    Resolve a `TableStyle` instance from a style name.

    This function looks up a style registry and returns a corresponding
    `TableStyle` object. It is used internally by `QuantTable` when a
    user passes `style="academic"` or similar.

    Parameters
    ----------
    style_name : str
        Name of the style to resolve. Typical values include
        'academic', 'minimal', 'boxed', etc., depending on the
        implementation.

    Returns
    -------
    TableStyle
        A style object representing the requested style.

    Raises
    ------
    KeyError
        If no style with the given name is registered.
    """
    key = style_name.lower()
    try:
        return _STYLE_REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown table style: {style_name!r}") from None