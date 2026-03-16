"""
betterreport
============

High-level tools to build publication-quality tables:

    import betterreport as br

    tbl = br.build_table(
        df,
        title="Performance statistics",
        caption="Annualized returns, volatilities and Sharpe ratios (2010-2020 daily data).",
        style="academic",
    )

    latex = tbl.to_latex()
"""

from .table_style import TableStyle, MetricFormat, get_table_style
from .table import ReportTable, MetricFormatLike
from typing import Any, Mapping, Literal
import pandas as pd

def build_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    caption: str | None = None,
    style: str = "academic",
    title_align: Literal["left", "center", "right"] = "center",
    caption_align: Literal["left", "center", "right"] = "left",
    title_position: Literal["above", "below"] = "above",
    caption_position: Literal["above", "below"] = "below",
    formats: Mapping[Any, MetricFormatLike] | None = None,
    format_axis: Literal["rows", "columns"] = "rows",
    note: str | None = None,
    style_options: Mapping[str, Any] | None = None,
    highlight_mode: Literal["max", "min"] | None = None,
    highlight_axis: Literal["rows", "columns"] = "rows",
    default_format: MetricFormatLike | None = MetricFormat(kind="number", decimals=2),
    significance_stat: Literal[None, "pvalue", "stderr", "tstat"] = None,
    significance_layout: Literal["stack", "inline"] = "stack",
    significance_thresholds: tuple[float, float, float] = (0.10, 0.05, 0.01)
) -> ReportTable:
    """
    API function to build a `ReportTable` from a pandas DataFrame.
    This encapsulates common parameters and conventions for
    rendering publication-quality tables.

    Table Style is based on a named preset (e.g. 'academic', 'minimal'),
    with optional overrides via `style_options`.

    Parameters
    ----------
    data : pandas.DataFrame
        Tabular data to render. By convention:
            - formatting can be applied per row or per column via
            `format_axis` and `formats`.
        The index is used as the stub column in the rendered
        table. Column labels appear in the header row.
    title : str or None, default None
        Optional title for the table. In LaTeX this is typically used as
        the table caption heading.
    caption : str or None, default None
        Optional descriptive text explaining the content of the table.
        In LaTeX and HTML this is usually rendered as a short paragraph
        immediately above or below the table.
    style : str, default 'academic'
        Name of the base style to use (e.g. 'academic', 'minimal').
        The name is resolved internally to a `TableStyle` via
        `get_table_style`.
    title_align : {'left', 'center', 'right'}, default 'center'
        Horizontal alignment of the title text.
    caption_align : {'left', 'center', 'right'}, default 'left'
        Horizontal alignment of the caption text.
    title_position : {'above', 'below'}, default 'above'
        Whether the title should be rendered above or below the table.
    caption_position : {'above', 'below'}, default 'below'
        Whether the caption should be rendered above or below the table.
    formats : mapping or None, default None
        Mapping from metric labels to `MetricFormat` objects. The keys
        refer either to row labels or to column labels, depending on
        `format_axis`. If a metric is not listed in this mapping,
        numeric values for that metric are rendered with a default
        representation.
    format_axis : {'rows', 'columns'}, default 'rows'
        Axis along which `formats` keys are interpreted:
            - 'rows'   : keys in `formats` are row labels (`data.index`),
            - 'columns': keys in `formats` are column labels (`data.columns`).
        If the specified label cannot be found on the chosen axis, the
        corresponding format is silently ignored.
    note : str or None, default None
        Optional footnote texts to append below the table.
        By default footnotes are rendered in a smaller font size and italics.
        The note is prefixed with a 'Note:' label.
    style_options : mapping or None, default None
        Optional overrides applied on top of the base style resolved
        from `style`. Keys correspond to fields of `TableStyle`.
    highlight_mode : {'max', 'min'} or None, default None
        If not None, enables automatic highlighting of the maximum or
        minimum value in each row or column (depending on `format_axis`).
        Highlighting method (e.g. bold, italic, color) is determined by the settings
        of the selected style.
    highlight_axis : {'rows', 'columns'}, default 'rows'
        Axis along which to apply the `highlight_mode` if enabled.
        If None, defaults to the same axis as `format_axis`.
    default_format : MetricFormatLike or None, default MetricFormat(kind="number", decimals=2)
        Default formatting configuration to use for numeric values when no
        specific format is found in `formats` for a given metric.
    significance_stat : {'pvalue', 'stderr', 'tstat'} or None, default None
        If not None, enables significance annotation based on the specified statistic.
        The statistic values must be provided in the DataFrame as additional
        columns with a specific naming convention (e.g. 'MetricName_pvalue' for p-values).
        Example:
            For a given column 'Coefficient', if the DataFrale contains column 'Coefficient_pvalue',
            setting `significance_stat='pvalue'` will give :
                - p-value <= 0.01 : 'Coefficient*** (p-value)'
                - 0.01 < p-value <= 0.05 : 'Coefficient** (p-value)'
                - 0.05 < p-value <= 0.10 : 'Coefficient* (p-value)'
                - p-value > 0.10 : 'Coefficient (p-value)'
            If the statistic is stderr or tstat if will be rendered as :
                - 'Coefficient (stderr or tstat)'
    significance_layout : {'stack', 'inline'}, default 'stack'
        Layout for rendering significance annotations when `significance_stat` is enabled.
        - 'stack': annotations are rendered in a separate line below the metric value
        - 'inline': annotations are rendered on the same line as the metric value
        Example:
            - stack : 0.25**
                      (0.03)
            - inline: 0.25** (0.03)
    significance_thresholds : tuple of 3 floats, default (0.10, 0.05, 0.01)
        Thresholds for significance annotation when `significance_stat='pvalue'`.
        The tuple should contain three values corresponding to the thresholds for
        '*', '**', and '***' annotations respectively. Default is (0.10, 0.05, 0.01).
        IF significance_stat is 'stderr' or 'tstat', these thresholds are ignored.

    Returns
    -------
    ReportTable
        The constructed `ReportTable` object, high-level representation of
        the table with metadata and styling information.

    Examples
    --------
    >>> import betterreport as br
    >>> tbl = br.build_table(
    ...     df,
    ...     title="Performance statistics",
    ...     caption="Annualized returns, volatilities and Sharpe ratios.",
    ...     style="academic",
    ... )
    >>> latex = tbl.to_latex()
    """
    return ReportTable.from_dataframe(
        df=df,
        title=title,
        caption=caption,
        style=style,
        title_align=title_align,
        caption_align=caption_align,
        title_position=title_position,
        caption_position=caption_position,
        formats=formats,
        format_axis=format_axis,
        note=note,
        style_options=style_options,
        highlight_mode=highlight_mode,
        highlight_axis=highlight_axis,
        default_format=default_format,
        significance_stat=significance_stat,
        significance_layout=significance_layout,
        significance_thresholds=significance_thresholds
    )

all = [
    "build_table",
    "ReportTable",
    "MetricFormat",
    "TableStyle"
    ]
