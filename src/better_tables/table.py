"""
High-level API for rendering tables.

This module defines a small, opinionated abstraction around `pandas.DataFrame`
to produce publication-grade tables (LaTeX and HTML) for reports.

The core idea is:

- Input: a `pandas.DataFrame` containing the raw data to be rendered, with index and column labels.
- Configuration: a high-level style name (`style="academic"`, etc.), optional
  layout parameters (title, caption, alignment ...) and formatting rules per metric.
- Output: LaTeX/HTML table code ready to be embedded in a report.

Only the `ReportTable` class is meant to be used directly by end-users. Other
classes such as `TableStyle` or `MetricFormat` are internal implementation
details, even though they are documented here for clarity.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any, Literal, TypeAlias
import pandas as pd
from pathlib import Path
import warnings
import numbers
import math

from .table_style import TableStyle, MetricFormat, get_table_style
from .utils.render_utils import (
    _escape_latex,
    _escape_html,
    _build_latex_colspec,
    _compile_latex_to_pdf
)

MetricFormatLike: TypeAlias = MetricFormat | Mapping[str, Any]
DEFAULT_METRIC_FORMAT = MetricFormat(kind="number", decimals=2)

@dataclass
class ReportTable:
    """
    High-level representation of a report table.

    `ReportTable` encapsulates a `pandas.DataFrame` together with metadata
    (title, caption) and styling/formatting information, and can render
    the table to LaTeX or HTML.

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
    """

    data: pd.DataFrame
    title: str | None = None
    caption: str | None = None
    style_name: str = "academic"
    style: TableStyle | None = None
    title_align: Literal["left", "center", "right"] = "center"
    caption_align: Literal["left", "center", "right"] = "left"
    title_position: Literal["above", "below"] = "above"
    caption_position: Literal["above", "below"] = "below"
    formats: Mapping[Any, MetricFormatLike] | None = None
    format_axis: Literal["rows", "columns"] = "rows"
    note: str | None = None
    style_options: Mapping[str, Any] | None = None
    highlight_mode: Literal["max", "min"] | None = None
    highlight_axis: Literal["rows", "columns"] = "rows"
    default_format: MetricFormatLike | None = None
    significance_stat: Literal[None, "pvalue", "stderr", "tstat"] = None
    significance_layout: Literal["stack", "inline"] = "stack"
    significance_thresholds: tuple[float, float, float] = (0.10, 0.05, 0.01)

    @classmethod
    def from_dataframe(
        cls,
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
        significance_thresholds: tuple[float, float, float] = (0.10, 0.05, 0.01),
    ) -> ReportTable:
        """
        Typical constructor/factory method for `ReportTable`.
        User-friendly entry point.

        Parameters
        ----------
        Same as in `ReportTable` constructor.

        Returns
        -------
        ReportTable
            The constructed `ReportTable` object, high-level representation of
            the table with metadata and styling information.
        """
        data = df.copy()

        normalized_formats = cls._normalize_formats_static(formats)
        normalized_default_format = cls._coerce_metric_format_static(default_format) if default_format is not None else None

        obj = cls(
            data=data,
            title=title,
            caption=caption,
            style_name=style,
            style=None,
            title_align=title_align,
            caption_align=caption_align,
            title_position=title_position,
            caption_position=caption_position,
            formats=normalized_formats,
            format_axis=format_axis,
            style_options=style_options,
            note=note,
            highlight_mode=highlight_mode,
            highlight_axis=highlight_axis,
            default_format=normalized_default_format,
            significance_stat=significance_stat,
            significance_layout=significance_layout,
            significance_thresholds=significance_thresholds,
        )
        obj._validate_dataframe()
        obj._resolve_style()
        obj._warn_for_unused_formats()

        return obj

    # ------------------------------------------------------------------
    # |                     Rendering methods                          |
    # ------------------------------------------------------------------

    def to_latex(
        self,
        save_as_tex: bool = False,
        save_as_pdf: bool = False,
        path: str | Path | None = None,
        latex_engine: str | None = None,
        float_env: bool = True,
    ) -> str:
        """
        Render the table as LaTeX code.

        Parameters
        ----------
        save_as_tex : bool, default False
            If True, the LaTeX code is also saved to a .tex file at
            the location specified by `path`.
        save_as_pdf : bool, default False
            If True, the LaTeX code is compiled to PDF using the specified
            LaTeX engine and saved at the location specified by `path`.
            Implies `save_as_tex=True`.
        path : str or Path or None, default None
            If `save_as_tex` or `save_as_pdf` is True, the target file
            path where the .tex (and optionally .pdf) file will be saved.
            If None, defaults to 'report_table.tex' or 'report_table.pdf'
            in the current working directory.
        latex_engine : str, default 'pdflatex'
            LaTeX engine to use for PDF compilation (e.g. 'pdflatex',
            'xelatex', 'lualatex'). Only relevant if `save_as_pdf` is True.
            User must ensure that the specified engine is installed and
            available in the system PATH.
            Download path : https://miktex.org/download
            Then make sure to add it to your PATH.
            A typical path on Windows is:
                'C:\\Users\\<username>\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe'
        float_env : bool, default True
            If True, wrap the table in a floating `table` environment
            (the current default behaviour). If False, render a non-floating
            fragment suitable for embedding inside a minipage or custom
            layout. In that case, the content is centered and the caption
            is emitted via `\\captionof{table}{...}`.

        Returns
        -------
        str
            LaTeX representation of the table
        """
        self._validate_dataframe()
        self._resolve_style()
        display_data, display_aux = self._prepare_display_frame()
        formatted = self._build_formatted_frame(display_data=display_data, display_aux=display_aux)
        highlight_mask = self._compute_highlight_mask(display_data)
        body = self._build_latex_body(formatted, highlight_mask=highlight_mask)

        n_stub_cols = self._get_stub_frame(display_data.index).shape[1]
        colspec = _build_latex_colspec(
            n_body_cols=n_stub_cols + formatted.shape[1],
            style=self.style,
        )

        lines: list[str] = []

        if float_env:
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering")
        else:
            lines.append(r"\begin{center}")

        styled_title = self._format_title_for_caption(self.title) if self.title else None
        cap: str | None = None
        if styled_title or self.caption:
            if styled_title and self.caption:
                cap = (
                    styled_title
                    + r"\\[0.4em]"
                    + _escape_latex(self.caption)
                )
            elif styled_title:
                cap = styled_title
            else:
                cap = _escape_latex(self.caption)

        if cap is not None:
            if float_env:
                lines.append(r"\caption{" + cap + r"}")
            else:
                lines.append(r"\captionof{table}{" + cap + r"}")

        lines.append(r"\vspace{0.4em}")
        lines.append(r"\begin{tabular}{" + colspec + r"}")
        lines.append(body)
        lines.append(r"\end{tabular}")

        if self.note:
            lines.append(r"\vspace*{1.2em}")
            lines.append(
                r"\parbox{0.9\linewidth}{"
                r"\raggedright\footnotesize"
                r"\emph{\underline{Note:}~" + _escape_latex(self.note) + r"}}"
            )

        if float_env:
            lines.append(r"\end{table}")
        else:
            lines.append(r"\end{center}")

        latex_code = "\n".join(lines)

        if save_as_tex or save_as_pdf:
            if path is None:
                path = Path("report_table.tex")
            else:
                path = Path(path)

            tex_path = path.with_suffix(".tex")

            wrapper = (
                "\\documentclass{article}\n"
                "\\usepackage{booktabs}\n"
                "\\usepackage[margin=2.5cm]{geometry}\n"
                "\\usepackage{caption}\n"
                "\\usepackage[table]{xcolor}\n"
                "\\usepackage{makecell}\n"
                "\\captionsetup[table]{labelformat=empty}\n"
                "\\begin{document}\n\n"
                f"{latex_code}\n\n"
                "\\end{document}\n"
            )
            tex_path.write_text(wrapper, encoding="utf-8")

            if save_as_pdf:
                pdf_path = _compile_latex_to_pdf(
                    tex_path=tex_path,
                    engine=latex_engine
                )
                if pdf_path != path.with_suffix(".pdf"):
                    pdf_path.rename(path.with_suffix(".pdf"))
        
        return latex_code

    
    def to_html(self) -> str:
        """
        Render the table as HTML code.

        The returned string is a self-contained HTML fragment containing
        a `<table>` element.
        Can be embedded directly into an HTML document or further processed
        as needed.

        Returns
        -------
        str
            HTML representation of the table.
        """
        self._validate_dataframe()
        self._resolve_style()
        display_data, display_aux = self._prepare_display_frame()
        formatted = self._build_formatted_frame(display_data=display_data, display_aux=display_aux)
        highlight_mask = self._compute_highlight_mask(display_data)
        table_html = self._build_html_body(formatted, highlight_mask=highlight_mask)

        blocks: list[str] = []

        if self.title and self.title_position == "above":
            align_class = f"br-align-{self.title_align}"
            blocks.append(
                f'<div class="br-table-title {align_class}">'
                f"{_escape_html(self.title)}</div>"
            )

        if self.caption and self.caption_position == "above":
            align_class = f"br-align-{self.caption_align}"
            blocks.append(
                f'<div class="br-table-caption {align_class}">'
                f"{_escape_html(self.caption)}</div>"
            )

        blocks.append(table_html)

        if self.caption and self.caption_position == "below":
            align_class = f"br-align-{self.caption_align}"
            blocks.append(
                f'<div class="br-table-caption {align_class}">'
                f"{_escape_html(self.caption)}</div>"
            )

        if self.title and self.title_position == "below":
            align_class = f"br-align-{self.title_align}"
            blocks.append(
                f'<div class="br-table-title {align_class}">'
                f"{_escape_html(self.title)}</div>"
            )

        return "\n".join(blocks)

    # ------------------------------------------------------------------
    # |                       Internal helpers                         |
    # ------------------------------------------------------------------

    def _validate_dataframe(self) -> None:
        """
        Validate the input DataFrame.

        This method is responsible for basic sanity checks on the `data`
        attribute, such as ensuring the index and column labels are
        suitable for rendering and that there are no unsupported dtypes.

        Raises
        ------
        ValueError
            If the DataFrame structure is not compatible with the table
            rendering logic.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("ReportTable.data must be a pandas.DataFrame")

        if self.data.empty:
            raise ValueError("ReportTable.data must not be empty")

        try:
            _ = [str(idx) for idx in self.data.index]
            if isinstance(self.data.columns, pd.MultiIndex):
                for col in self.data.columns:
                    _ = [str(x) for x in col]
            else:
                _ = [str(col) for col in self.data.columns]
        except Exception as exc:
            raise ValueError("Index/column labels must be convertible to string") from exc

        if self.format_axis not in ("rows", "columns"):
            raise ValueError("format_axis must be 'rows' or 'columns'")

        if self.highlight_axis not in ("rows", "columns"):
            raise ValueError("highlight_axis must be 'rows' or 'columns'")

        if self.significance_stat not in (None, "pvalue", "stderr", "tstat"):
            raise ValueError("significance_stat must be None, 'pvalue', 'stderr' or 'tstat'")

        if self.significance_layout not in ("stack", "inline"):
            raise ValueError("significance_layout must be 'stack' or 'inline'")

        if len(self.significance_thresholds) != 3:
            raise ValueError("significance_thresholds must contain exactly 3 thresholds")

    def _resolve_style(self) -> None:
        """
        Resolve and apply the effective table style.

        This method combines:

        - the base style resolved from `style_name` via `get_table_style`,
        - any overrides provided in `style_options`,

        and stores the result in `self.style`.
        """
        base = get_table_style(self.style_name)

        eff = base.with_overrides(self.style_options)
        self.style = eff

    def _format_title_for_caption(self, title: str) -> str:
        """
        Format the title string for inclusion in a LaTeX caption.

        Parameters
        ----------
        title : str
            Title text to format.

        Returns
        -------
        str
            LaTeX-formatted title string, with bold and/or underline
            applied according to style settings.
        """
        txt = _escape_latex(title)

        bold = self.style.title_bold
        underline = self.style.title_underline

        if bold and underline:
            return r"\textbf{\underline{" + txt + "}}"
        elif bold:
            return r"\textbf{" + txt + "}"
        elif underline:
            return r"\underline{" + txt + "}"
        else:
            return txt
    
    def _get_metric_format(self, label: Any) -> MetricFormat | None:
        """
        Retrieve the formatting configuration for a given metric label.

        Parameters
        ----------
        label : str
            Metric label to look up. This can be either a row label or
            a column label depending on `format_axis`.

        Returns
        -------
        MetricFormat or None
            The corresponding `MetricFormat` if one is defined for the
            given label, otherwise None.
        """
        if not self.formats:
            return None
        if label in self.formats:
            return self.formats[label]
        label_str = str(label)
        if label_str in self.formats:
            return self.formats[label_str]

        return None

    def _warn_for_unused_formats(self) -> None:
        """
        Warn the user if there are any keys in `formats` that do not correspond to
        any row/column label in the DataFrame.
        """
        if not self.formats:
            return

        valid_labels = self.data.index if self.format_axis == "rows" else self.data.columns
        valid_raw = set(valid_labels)
        valid_str = {str(x) for x in valid_labels}

        unknown = [
            key for key in self.formats
            if key not in valid_raw and str(key) not in valid_str
        ]

        if unknown:
            warnings.warn(
                f"Some format keys do not match any {self.format_axis[:-1]} label and were ignored: "
                + ", ".join(repr(k) for k in unknown),
                category=UserWarning,
                stacklevel=2,
            )

    def _format_value(self, label: Any, value: Any) -> str:
        """
        Format a single cell value according to metric-specific rules.

        This method applies the following logic:
            1. If the value is a string, return it as-is.
            2. If a `MetricFormat` is defined for the given label, use it to format the value.
            3. If the value is None or NaN, return the `na_rep` string from the style.
            4. For numeric values, apply a default formatting (integers as-is, floats with 2 decimals).
            5. For other types, convert to string using `str()`.

        Parameters
        ----------
        label : str
            Metric label associated with this value (row or column).
        value : Any
            Raw value to format.

        Returns
        -------
        str
            Formatted string representation of the value.
        """
        if isinstance(value, str):
            return value

        fmt = self._get_metric_format(label)
        if fmt is not None:
            return fmt.format(value)
    
        if value is None:
            return self.style.na_rep

        try:
            if isinstance(value, bool):
                return str(value)

            if isinstance(value, numbers.Integral):
                return str(int(value))

            if isinstance(value, numbers.Real):
                v = float(value)
                if math.isnan(v):
                    return self.style.na_rep
                default_fmt = self.default_format
                return default_fmt.format(v)

            return str(value)
        except Exception:
            return str(value)

    def _prepare_display_frame(self) -> tuple[pd.DataFrame, dict[Any, pd.Series]]:
        """
        Prepare the display frame and auxiliary significance series.

        If significance rendering is enabled, columns of the form:
            X + X_<significance_stat>
        are merged into a single rendered column X.

        This works for both flat columns and MultiIndex columns:
        - flat example:            beta + beta_pvalue
        - MultiIndex example:      ('Model 1', 'beta') + ('Model 1', 'beta_pvalue')

        Returns
        -------
        tuple[pd.DataFrame, dict[Any, pd.Series]]
            The display dataframe and an auxiliary mapping from main column labels
            to the corresponding auxiliary statistic series.
        """
        if self.significance_stat is None:
            return self.data.copy(), {}

        suffix = f"_{self.significance_stat}"
        columns = list(self.data.columns)

        aux_map: dict[Any, pd.Series] = {}
        kept_columns: list[Any] = []

        if isinstance(self.data.columns, pd.MultiIndex):
            column_set = set(columns)

            for col in columns:
                # MultiIndex column labels are tuples
                last = str(col[-1])

                # Skip auxiliary columns themselves
                if last.endswith(suffix):
                    continue

                kept_columns.append(col)

                aux_col = col[:-1] + (f"{last}{suffix}",)
                if aux_col in column_set:
                    aux_map[col] = self.data[aux_col]

            display_data = self.data[kept_columns].copy()
            return display_data, aux_map

        # Flat columns
        for col in columns:
            col_str = str(col)
            if col_str.endswith(suffix):
                continue

            kept_columns.append(col)

            aux_name = f"{col_str}{suffix}"
            aux_col = next((candidate for candidate in columns if str(candidate) == aux_name), None)
            if aux_col is not None:
                aux_map[col] = self.data[aux_col]

        display_data = self.data[kept_columns].copy()
        return display_data, aux_map

    def _format_significance_stars(self, pvalue: Any) -> str:
        """
        Format significance stars based on a p-value.

        Stars are only computed when `significance_stat == 'pvalue'`.
        """
        if self.significance_stat != "pvalue":
            return ""

        if pvalue is None or pd.isna(pvalue):
            return ""

        try:
            p = float(pvalue)
        except Exception:
            return ""

        t1, t2, t3 = self.significance_thresholds
        if p <= t3:
            return "***"
        if p <= t2:
            return "**"
        if p <= t1:
            return "*"
        return ""

    def _format_aux_value(self, value: Any) -> str:
        """
        Format the auxiliary significance statistic displayed in parentheses.
        """
        if value is None or pd.isna(value):
            return self.style.na_rep

        try:
            v = float(value)
        except Exception:
            return str(value)

        return MetricFormat(kind="number", decimals=3).format(v)

    def _build_formatted_frame(self, display_data: pd.DataFrame | None = None, display_aux: dict[Any, pd.Series] | None = None) -> pd.DataFrame:
        """
        Build a fully formatted DataFrame of strings.

        This method converts the original `data` into a new DataFrame
        where all cells are string representations, after applying:

        - NA handling according to `self.style.na_rep`,
        - metric-specific formatting via `_format_value`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the same shape as `data`, but with string
            values ready to be inserted into LaTeX/HTML templates.
        """
        if display_data is None:
            display_data = self.data
        if display_aux is None:
            display_aux = {}

        formatted = pd.DataFrame(index=display_data.index, columns=display_data.columns, dtype=object)

        for row_label in display_data.index:
            for col_label in display_data.columns:
                raw = display_data.loc[row_label, col_label]

                if pd.isna(raw):
                    main_text = self.style.na_rep
                else:
                    metric_label: Any
                    if self.format_axis == "rows":
                        metric_label = row_label
                    else:
                        metric_label = col_label

                    main_text = self._format_value(metric_label, raw)

                if col_label in display_aux:
                    aux_raw = display_aux[col_label].loc[row_label]
                    aux_text = self._format_aux_value(aux_raw)
                    stars = self._format_significance_stars(aux_raw)

                    if self.significance_layout == "stack":
                        text = f"{main_text}{stars}\n({aux_text})"
                    else:
                        text = f"{main_text}{stars} ({aux_text})"
                else:
                    text = main_text

                formatted.loc[row_label, col_label] = text

        return formatted.astype(str)

    def _get_stub_frame(self, index: pd.Index | pd.MultiIndex | None = None) -> pd.DataFrame:
        """
        Build the left stub area from the DataFrame index.

        For a simple Index, this produces one stub column.
        For a MultiIndex, it produces one stub column per level and sparsifies
        repeated values vertically.
        """
        idx = self.data.index if index is None else index

        if not isinstance(idx, pd.MultiIndex):
            name = "" if idx.name is None else str(idx.name)
            return pd.DataFrame({name: [str(x) for x in idx]}, index=idx)

        names = ["" if name is None else str(name) for name in idx.names]
        stub = pd.DataFrame(idx.tolist(), index=idx, columns=names).astype(str)
        return self._sparsify_stub_frame(stub)

    @staticmethod
    def _sparsify_stub_frame(stub: pd.DataFrame) -> pd.DataFrame:
        """
        Sparsify a stub frame by removing repeated contiguous labels
        level by level.

        Important:
        comparison must be performed against the original previous row,
        not against the already sparsified output, otherwise labels may
        reappear every other row.
        """
        original = stub.copy()
        out = stub.copy()

        for i in range(1, len(original)):
            prev = list(original.iloc[i - 1])
            curr = list(original.iloc[i])

            for level in range(len(curr)):
                if all(curr[j] == prev[j] for j in range(level + 1)):
                    out.iat[i, level] = ""
                else:
                    break

        return out

    def _get_header_rows(self, columns: pd.Index | pd.MultiIndex) -> list[list[str]]:
        """
        Build header rows from the DataFrame columns.

        For a simple Index, returns a single header row.
        For a MultiIndex, returns one row per level.
        """
        if not isinstance(columns, pd.MultiIndex):
            return [[str(c) for c in columns]]

        levels = columns.nlevels
        tuples = list(columns)

        rows: list[list[str]] = []
        for level in range(levels):
            rows.append([str(t[level]) for t in tuples])

        return rows

    @staticmethod
    def _compress_runs(labels: list[str]) -> list[tuple[str, int]]:
        """
        Compress consecutive identical labels into (label, span) pairs.
        """
        if not labels:
            return []

        runs: list[tuple[str, int]] = []
        current = labels[0]
        count = 1

        for label in labels[1:]:
            if label == current:
                count += 1
            else:
                runs.append((current, count))
                current = label
                count = 1

        runs.append((current, count))
        return runs

    def _build_latex_body(self, formatted: pd.DataFrame, highlight_mask: pd.DataFrame | None = None) -> str:
        """
        Build the LaTeX body of the table (rows only).

        Given a fully formatted DataFrame, this method is responsible
        for generating the LaTeX code corresponding to the header row,
        body rows, and any horizontal rules, according to `self.style`.

        Parameters
        ----------
        formatted : pandas.DataFrame
            Fully formatted DataFrame of strings as returned by
            `_build_formatted_frame`.
        highlight_mask : pandas.DataFrame or None
            Optional boolean mask indicating which cells should be highlighted.
            If provided, must have the same shape and index/columns as `formatted`.

        Returns
        -------
        str
            LaTeX code for the table body (excluding caption, title,
            and outer `table` environment).
        """
        s = self.style
        lines: list[str] = []

        if s.outer_border in ("box", "top_bottom"):
            lines.append(r"\hline")

        stub = self._get_stub_frame(formatted.index)
        header_rows = self._get_header_rows(formatted.columns)

        for level_idx, header_labels in enumerate(header_rows):
            row_cells: list[str] = []

            if level_idx == len(header_rows) - 1:
                for stub_name in stub.columns:
                    text = _escape_latex(str(stub_name))
                    if s.bold_header:
                        text = r"\textbf{" + text + "}"
                    row_cells.append(text)
            else:
                row_cells.extend([""] * len(stub.columns))

            runs = self._compress_runs(header_labels)

            for run_idx, (label, span) in enumerate(runs):
                label_text = label

                if (
                    isinstance(formatted.columns, pd.MultiIndex)
                    and level_idx == 0
                    and run_idx < len(runs) - 1
                ):
                    label_text = f"{label_text}"

                text = _escape_latex(label_text)
                if s.bold_header:
                    text = r"\textbf{" + text + "}"

                if span > 1:
                    row_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{text}}}")
                else:
                    row_cells.append(text)

            lines.append(" & ".join(row_cells) + r" \\")

        if s.inner_hlines in ("all", "header", "header_footer"):
            lines.append(r"\hline")

        n_rows = len(formatted.index)
        for i, idx in enumerate(formatted.index):
            row_cells: list[str] = []

            for stub_col in stub.columns:
                idx_text = _escape_latex(str(stub.loc[idx, stub_col]))
                if s.bold_index and idx_text != "":
                    idx_text = r"\textbf{" + idx_text + "}"
                row_cells.append(idx_text)

            for col in formatted.columns:
                cell_raw = str(formatted.loc[idx, col])
                cell_text = self._latexify_cell_text(cell_raw)

                if highlight_mask is not None and bool(highlight_mask.loc[idx, col]):
                    cell_text = self._apply_latex_highlight(cell_text)

                if s.last_row_emphasis and i == n_rows - 1:
                    cell_text = r"\textbf{" + cell_text + "}"
                row_cells.append(cell_text)

            lines.append(" & ".join(row_cells) + r" \\")

            is_last = (i == n_rows - 1)
            if s.inner_hlines == "all" and not is_last:
                lines.append(r"\hline")

            if s.last_row_emphasis and not is_last and i == n_rows - 2:
                lines.append(r"\hline")

        need_bottom = False
        if s.outer_border in ("box", "top_bottom"):
            need_bottom = True
        if s.inner_hlines == "header_footer":
            need_bottom = True

        if need_bottom:
            lines.append(r"\hline")

        return "\n".join(lines)

    def _build_html_body(self, formatted: pd.DataFrame, highlight_mask: pd.DataFrame | None = None) -> str:
        """
        Build the HTML body of the table (rows only).

        Given a fully formatted DataFrame, this method is responsible
        for generating the HTML `<table>` structure, including `<thead>`
        and `<tbody>` sections, as well as CSS classes consistent with
        the current style.

        Parameters
        ----------
        formatted : pandas.DataFrame
            Fully formatted DataFrame of strings as returned by
            `_build_formatted_frame`.
        highlight_mask : pandas.DataFrame or None
            Optional boolean mask indicating which cells should be highlighted.
            If provided, must have the same shape and index/columns as `formatted`.

        Returns
        -------
        str
            HTML code for the `<table>` element (excluding external
            wrappers such as title blocks or footnotes).
        """
        s = self.style

        table_classes = ["br-table", f"br-style-{_escape_html(s.name)}"]
        table_class_attr = " ".join(table_classes)

        header_align_css = s.header_align
        body_align_css = s.body_align

        stub = self._get_stub_frame(formatted.index)
        header_rows = self._get_header_rows(formatted.columns)

        thead_lines: list[str] = []
        thead_lines.append("<thead>")

        for level_idx, header_labels in enumerate(header_rows):
            thead_lines.append("<tr>")

            if level_idx == len(header_rows) - 1:
                for stub_name in stub.columns:
                    text = _escape_html(str(stub_name))
                    if s.bold_header:
                        text = f"<strong>{text}</strong>"
                    thead_lines.append(
                        f'<th scope="col" style="text-align:{header_align_css};">{text}</th>'
                    )
            else:
                for _ in stub.columns:
                    thead_lines.append('<th scope="col"></th>')

            runs = self._compress_runs(header_labels)

            for run_idx, (label, span) in enumerate(runs):
                label_text = label

                if (
                    isinstance(formatted.columns, pd.MultiIndex)
                    and level_idx == 0
                    and run_idx < len(runs) - 1
                ):
                    label_text = f"{label_text}"

                text = _escape_html(label_text)
                if s.bold_header:
                    text = f"<strong>{text}</strong>"
                colspan = f' colspan="{span}"' if span > 1 else ""
                thead_lines.append(
                    f'<th scope="col"{colspan} style="text-align:{header_align_css};">{text}</th>'
                )

            thead_lines.append("</tr>")

        thead_lines.append("</thead>")

        tbody_lines: list[str] = []
        tbody_lines.append("<tbody>")

        n_rows = len(formatted.index)
        for i, idx in enumerate(formatted.index):
            row_classes: list[str] = []
            if s.zebra_striping:
                row_classes.append("br-row-even" if i % 2 == 0 else "br-row-odd")
            if s.last_row_emphasis and i == n_rows - 1:
                row_classes.append("br-row-last")

            row_class_attr = f' class="{" ".join(row_classes)}"' if row_classes else ""
            tbody_lines.append(f"<tr{row_class_attr}>")

            for stub_col in stub.columns:
                idx_text = _escape_html(str(stub.loc[idx, stub_col]))
                if s.bold_index and idx_text != "":
                    idx_text = f"<strong>{idx_text}</strong>"
                tbody_lines.append(
                    f'<th scope="row" style="text-align:left;">{idx_text}</th>'
                )

            for col in formatted.columns:
                cell_raw = str(formatted.loc[idx, col])
                parts = [_escape_html(part) for part in cell_raw.split("\n")]
                cell_text = "<br>".join(parts)

                if highlight_mask is not None and bool(highlight_mask.loc[idx, col]):
                    cell_text = self._apply_html_highlight(cell_text)
                if s.last_row_emphasis and i == n_rows - 1:
                    cell_text = f"<strong>{cell_text}</strong>"
                tbody_lines.append(
                    f'<td style="text-align:{body_align_css};">{cell_text}</td>'
                )

            tbody_lines.append("</tr>")

        tbody_lines.append("</tbody>")

        html_parts = [
            f'<table class="{table_class_attr}">',
            *thead_lines,
            *tbody_lines,
            "</table>",
        ]

        return "\n".join(html_parts)

    @staticmethod
    def _coerce_metric_format_static(value: MetricFormatLike) -> MetricFormat:
        """
        Coerce a value to a MetricFormat instance.

        This static method is used to convert entries in the `formats` mapping to `MetricFormat` instances
        if they are provided as dictionaries.

        Parameters
        ----------
        value : MetricFormat or mapping
            The value to coerce. It can be either a `MetricFormat` instance or a mapping with keys
            corresponding to the fields of `MetricFormat`.

        Returns
        -------
        MetricFormat
            The coerced `MetricFormat` instance.
        """
        if isinstance(value, MetricFormat):
            return value

        if isinstance(value, Mapping):
            allowed_keys = {"kind", "percent_input", "decimals", "formatter"}
            unknown_keys = sorted(set(value) - allowed_keys)
            if unknown_keys:
                raise ValueError(
                    "Invalid MetricFormat config key(s): "
                    + ", ".join(repr(k) for k in unknown_keys)
                )
            return MetricFormat(**value)

        raise TypeError(
            "Each entry in formats must be either a MetricFormat or a mapping "
            "compatible with MetricFormat."
        )

    @staticmethod
    def _normalize_formats_static(formats: Mapping[Any, MetricFormatLike] | None) -> dict[Any, MetricFormat] | None:
        """
        Normalize the `formats` mapping by coercing all values to `MetricFormat` instances.

        Parameters
        ----------
        formats : mapping or None
            Original formats mapping provided by the user, where values can be either `MetricFormat`
            instances or dictionaries.

        Returns
        -------
        dict or None
            Normalized formats mapping where all values are `MetricFormat` instances, or None if
            the input was None.
        """
        if not formats:
            return None

        return {
            key: ReportTable._coerce_metric_format_static(value)
            for key, value in formats.items()
        }

    def _compute_highlight_mask(self, display_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Compute a boolean mask indicating which cells should be highlighted based on the current
        `highlight` configuration.

        The mask is a DataFrame of the same shape as `self.data`, with True for cells that should
        be highlighted and False otherwise.

        Returns
        -------
        pandas.DataFrame
            Boolean mask indicating highlighted cells.
        """
        if display_data is None:
            display_data = self.data

        mask = pd.DataFrame(False, index=display_data.index, columns=display_data.columns)

        if self.highlight_mode is None:
            return mask

        axis = 1 if self.highlight_axis == "rows" else 0

        if axis == 1:
            for idx in display_data.index:
                row = pd.to_numeric(display_data.loc[idx], errors="coerce")
                valid = row.dropna()
                if valid.empty:
                    continue

                target = valid.max() if self.highlight_mode == "max" else valid.min()
                selected = row.eq(target) & row.notna()
                mask.loc[idx, selected.index[selected]] = True
        else:
            for col in display_data.columns:
                column = pd.to_numeric(display_data[col], errors="coerce")
                valid = column.dropna()
                if valid.empty:
                    continue

                target = valid.max() if self.highlight_mode == "max" else valid.min()
                selected = column.eq(target) & column.notna()
                mask.loc[selected.index[selected], col] = True

        return mask
    
    def _apply_latex_highlight(self, text: str) -> str:
        """
        Apply LaTeX highlighting styles to the given text based on the current style settings.

        Parameters
        ----------
        text : str
            The text to which highlighting styles should be applied.
        
        Returns
        -------
        str
            The input text wrapped in LaTeX commands that apply the appropriate styles (e.g., bold, italic, color).
            If no highlighting styles are set, the original text is returned unchanged.
        """
        s = self.style
        out = text

        if s.highlight_color:
            out = rf"\textcolor{{{s.highlight_color}}}{{{out}}}"
        if s.highlight_underline:
            out = rf"\underline{{{out}}}"
        if s.highlight_italic:
            out = rf"\textit{{{out}}}"
        if s.highlight_bold:
            out = rf"\textbf{{{out}}}"

        return out

    def _apply_html_highlight(self, text: str) -> str:
        """
        Apply HTML highlighting styles to the given text based on the current style settings.

        Parameters
        ----------
        text : str
            The text to which highlighting styles should be applied.

        Returns
        -------
        str
            The input text wrapped in HTML tags that apply the appropriate styles (e.g., bold, italic, color).
            If no highlighting styles are set, the original text is returned unchanged.
        """
        s = self.style
        styles: list[str] = []

        if s.highlight_bold:
            styles.append("font-weight:700")
        if s.highlight_italic:
            styles.append("font-style:italic")
        if s.highlight_underline:
            styles.append("text-decoration:underline")
        if s.highlight_color:
            styles.append(f"color:{s.highlight_color}")

        if not styles:
            return text

        return f'<span style="{"; ".join(styles)}">{text}</span>'

    def _latexify_cell_text(self, text: str) -> str:
        """
        Convert a formatted cell text to valid LaTeX.

        If the text contains line breaks, wrap it in \\makecell so that the
        line breaks stay inside the same table cell.

        Parameters
        ----------
        text : str
            The formatted cell text to convert.
        
        Returns
        -------
        str
            LaTeX code representing the cell content, with proper escaping and line break handling.
        """
        parts = text.split("\n")

        if len(parts) == 1:
            return _escape_latex(parts[0])

        escaped_parts = [_escape_latex(part) for part in parts]
        return r"\makecell[r]{" + r" \\ ".join(escaped_parts) + "}"