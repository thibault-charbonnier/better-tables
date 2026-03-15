"""
High-level API for rendering tables.

This module defines a small, opinionated abstraction around `pandas.DataFrame`
to produce publication-grade tables (LaTeX and HTML) for reports.

The core idea is:

- Input: a `pandas.DataFrame` where rows and columns represent metrics and
  portfolios/strategies (or the other way around).
- Configuration: a high-level style name (`style="academic"`, etc.), optional
  layout parameters (title, caption, alignment) and formatting rules per metric.
- Output: LaTeX/HTML table code ready to be embedded in a report.

Only the `ReportTable` class is meant to be used directly by end-users. Other
classes such as `TableStyle` or `MetricFormat` are internal implementation
details, even though they are documented here for clarity.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any, Literal
import pandas as pd
from pathlib import Path

from .table_style import TableStyle, MetricFormat, get_table_style
from .utils.render_utils import (
    _escape_latex,
    _escape_html,
    _build_latex_colspec,
    _compile_latex_to_pdf
)


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
    formats: Mapping[str, MetricFormat] | None = None
    format_axis: Literal["rows", "columns"] = "rows"
    note: str | None = None
    style_options: Mapping[str, Any] | None = None

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
        formats: Mapping[str, MetricFormat] | None = None,
        format_axis: Literal["rows", "columns"] = "rows",
        note: str | None = None,
        style_options: Mapping[str, Any] | None = None,
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
            formats=formats,
            format_axis=format_axis,
            style_options=style_options,
            note=note,
        )
        obj._validate_dataframe()
        obj._resolve_style()

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
        formatted = self._build_formatted_frame()
        body = self._build_latex_body(formatted)

        colspec = _build_latex_colspec(
            n_body_cols=formatted.shape[1],
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
        formatted = self._build_formatted_frame()
        table_html = self._build_html_body(formatted)

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

        if isinstance(self.data.index, pd.MultiIndex):
            raise ValueError("MultiIndex index is not supported in ReportTable")
        if isinstance(self.data.columns, pd.MultiIndex):
            raise ValueError("MultiIndex columns are not supported in ReportTable")

        try:
            _ = [str(idx) for idx in self.data.index]
            _ = [str(col) for col in self.data.columns]
        except Exception as exc:
            raise ValueError("Index/column labels must be convertible to string") from exc

        if self.format_axis not in ("rows", "columns"):
            raise ValueError("format_axis must be 'rows' or 'columns'")

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
    
    def _get_metric_format(self, label: str) -> MetricFormat | None:
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

    def _format_value(self, label: str, value: Any) -> str:
        """
        Format a single cell value according to metric-specific rules.

        This method applies the following logic:

        - If `value` is already a string, most implementations will
          return it unchanged.
        - Otherwise, it looks up a `MetricFormat` for the given metric
          label via `_get_metric_format` (using `format_axis` to decide
          whether labels refer to rows or columns).
        - If a `MetricFormat` is found, it is used to format the value.
        - If no format is found, a default numeric representation is used.

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
            return ""
        return f"{value}"

    def _build_formatted_frame(self) -> pd.DataFrame:
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
        formatted = pd.DataFrame(index=self.data.index, columns=self.data.columns, dtype=object)

        for row_label in self.data.index:
            for col_label in self.data.columns:
                raw = self.data.loc[row_label, col_label]

                if pd.isna(raw):
                    formatted.loc[row_label, col_label] = self.style.na_rep
                    continue

                metric_label: Any
                if self.format_axis == "rows":
                    metric_label = row_label
                else:
                    metric_label = col_label

                text = self._format_value(metric_label, raw)
                formatted.loc[row_label, col_label] = text

        return formatted.astype(str)

    def _build_latex_body(self, formatted: pd.DataFrame) -> str:
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

        headers: list[str] = [""]
        for col in formatted.columns:
            text = _escape_latex(str(col))
            if s.bold_header:
                text = r"\textbf{" + text + "}"
            headers.append(text)
        lines.append(" & ".join(headers) + r" \\")

        if s.inner_hlines in ("all", "header", "header_footer"):
            lines.append(r"\hline")

        n_rows = len(formatted.index)
        for i, idx in enumerate(formatted.index):
            row_cells: list[str] = []

            idx_text = _escape_latex(str(idx))
            if s.bold_index:
                idx_text = r"\textbf{" + idx_text + "}"
            row_cells.append(idx_text)

            for col in formatted.columns:
                cell_text = _escape_latex(str(formatted.loc[idx, col]))
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

    def _build_html_body(self, formatted: pd.DataFrame) -> str:
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

        thead_lines: list[str] = []
        thead_lines.append("<thead>")
        thead_lines.append("<tr>")

        thead_lines.append('<th scope="col"></th>')

        for col in formatted.columns:
            text = _escape_html(str(col))
            if s.bold_header:
                text = f"<strong>{text}</strong>"
            thead_lines.append(
                f'<th scope="col" style="text-align:{header_align_css};">{text}</th>'
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

            idx_text = _escape_html(str(idx))
            if s.bold_index:
                idx_text = f"<strong>{idx_text}</strong>"
            tbody_lines.append(
                f'<th scope="row" style="text-align:left;">{idx_text}</th>'
            )

            for col in formatted.columns:
                cell_text = _escape_html(str(formatted.loc[idx, col]))
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
