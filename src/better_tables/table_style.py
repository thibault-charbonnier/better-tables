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
import warnings


@dataclass
class MetricFormat:
    """
    Formatting configuration for a single metric (row or column).

    This object describes how to convert numeric values associated to a given
    metric into human-readable strings (number vs percent, number of decimals,
    custom formatter, etc.). It is used by `ReportTable` when formatting table
    cells.

    Parameters
    ----------
    kind : {'number', 'percent'}, default 'number'
        Type of rendering to apply.

        - 'number'  : standard numeric rendering
        - 'percent' : append a '%' suffix, and optionally multiply by 100
          depending on `percent_input`

    percent_input : {'ratio', 'percent'}, default 'ratio'
        Only used when `kind='percent'`.

        - 'ratio'   : interpret input as a proportion, e.g. 0.123 -> 12.3%
        - 'percent' : interpret input as already expressed in percent,
                      e.g. 12.3 -> 12.3%

    decimals : int or None, default 2
        Number of decimal places to display when formatting numeric values.
        If None, a default representation is used.

    formatter : callable or None, default None
        Optional custom formatter of the form `formatter(value) -> str`.
        When provided, this function is responsible for producing the final
        string representation of the value, and other formatting options
        are ignored for this metric.
    """

    kind: Literal["number", "percent"] = "number"
    percent_input: Literal["ratio", "percent"] = "ratio"
    decimals: int | None = 2
    formatter: Callable[[Any], str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("number", "percent"):
            raise ValueError("MetricFormat.kind must be 'number' or 'percent'")
        if self.percent_input not in ("ratio", "percent"):
            raise ValueError("MetricFormat.percent_input must be 'ratio' or 'percent'")

    def format(self, value: Any) -> str:
        """
        Format a single value according to this metric format.
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

        if self.kind == "percent" and self.percent_input == "ratio":
            v *= 100.0

        if self.decimals is None:
            txt = f"{v}"
        else:
            fmt = f".{self.decimals}f"
            txt = format(v, fmt)

        if self.kind == "percent":
            txt += "%"

        return txt


@dataclass
class TableStyle:
    """
    Internal representation of a table style.
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
    highlight_bold: bool = True
    highlight_italic: bool = False
    highlight_underline: bool = False
    highlight_color: str | None = None

    def with_overrides(self, options: Mapping[str, Any] | None) -> TableStyle:
        """
        Return a copy of this style with selected fields overridden.
        """
        if not options:
            return self

        valid_fields = set(self.__dataclass_fields__.keys())
        unknown_keys = sorted(set(options) - valid_fields)

        if unknown_keys:
            warnings.warn(
                "Unknown style_options key(s) ignored: "
                + ", ".join(repr(k) for k in unknown_keys),
                category=UserWarning,
                stacklevel=2,
            )

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
        highlight_bold=True,
        highlight_italic=False,
        highlight_underline=False,
        highlight_color=None,
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
        highlight_bold=True,
        highlight_italic=False,
        highlight_underline=False,
        highlight_color=None,
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
        highlight_bold=True,
        highlight_italic=False,
        highlight_underline=False,
        highlight_color=None,
    ),
}


def get_table_style(style_name: str) -> TableStyle:
    """
    Resolve a `TableStyle` instance from a style name.
    """
    key = style_name.lower()
    try:
        return _STYLE_REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown table style: {style_name!r}") from None