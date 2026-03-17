# better-tables

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/better-tables.svg)](https://pypi.org/project/better-tables/)

A lightweight Python package to render pandas DataFrames into clean **LaTeX** and **HTML** tables for research reports, dashboards and publication-ready documents.

- **Lightweight DataFrame wrapper**: a unified entry point and object to turn a DataFrame into a report-ready table

- **Research-oriented**: built for statistics, econometrics and regression-style outputs, with native significance support

- **MultiIndex-friendly**: smart handling of complex row and column MultiIndex tables

- **Style and formatting focused:** flexible formatting rules and advanced styling options for precise rendering control

![Intro-Table](https://raw.githubusercontent.com/thibault-charbonnier/better-tables/master/static/intro-table.png)

---

## Install

```bash
pip install better-tables
```

---

## Development (uv)

```bash
git clone https://github.com/thibault-charbonnier/better-tables.git
cd better-tables
uv sync
```

---

## Quickstart

```python
import pandas as pd
from better_tables import build_table

df = pd.DataFrame(...)

table = build_table(
    df=df,
    title="Performance Summary",
    caption="Illustrative performance metrics of our strategies (2020-2025).",
    style="academic",
    highlight_mode="max",
    highlight_axis="columns",
)

latex = table.to_latex()
html = table.to_html()
```

---

## Core features

### 1) Clean rendering from `DataFrame`

`better-tables` is built around one high-level abstraction: `ReportTable`.

You start from a `pandas.DataFrame`, add optional metadata and formatting rules, and render the result to LaTeX or HTML.

---

### 2) Per-metric formatting

Formatting can be applied by **rows** or by **columns**.

Supported formatting options include:

- plain numeric formatting
- percentage formatting
- custom decimal precision
- fully custom formatter functions

```python
formats = {
    "Return": {"kind": "percent", "percent_input": "ratio", "decimals": 2},
    "Volatility": {"kind": "percent", "percent_input": "ratio", "decimals": 1},
    "Volume": {"kind": "number", "formatter": lambda v: f"{int(v):,}"},
}
format_axis = "rows" # "rows" or "columns"
```

#### Percent semantics

Percentage formatting is explicit:

- `kind="percent"` means the output should display `%`
- `percent_input="ratio"` means `0.123 -> 12.3%`
- `percent_input="percent"` means `12.3 -> 12.3%`

**Note:** you can use a dedicated `MetricFormat` object instead of a raw dictionary, with the same arguments.`

```python
MetricFormat(
    kind="number",              # "number" or "percent"
    percent_input="ratio",      # "ratio" or "percent"
    decimals=2,
    formatter=None,
)

# Both are accepted :
formats = {
    "Return": MetricFormat(kind="percent", percent_input="ratio", decimals=3),
    "Volatility": {"kind": "percent", "percent_input": "ratio", "decimals": 1}
}
```

---

### 3) Conditional highlighting

Highlight the maximum or minimum values by row or by column.

The visual appearance of the highlight is controlled by the selected style (see the dedicated section below).

```python
table = build_table(
    df=df,
    highlight_mode="max",
    highlight_axis="columns",
)
```

---

### 4) Significance rendering for statistics/econometrics tables

`better-tables` supports a compact significance workflow for econometrics and regression-style outputs, driven by the following parameters:

- `significance_stat=None | "pvalue" | "stderr" | "tstat"` : significance statistic to look for and put between parenthesis
- `significance_layout="stack" | "inline"` : whether to place the value in parentheses directly after the coefficient (`inline`) or on a new line below it (`stack`)`
- `significance_thresholds=(0.10, 0.05, 0.01)` : pvalue thresholds for significance stars

We use the following simple convention :

- If significance_stat is not `None` then for every column `col`, the code will look for `col_{stat}` where `stat` is the selected statistic (`"pvalue" | "stderr" | "tstat"`) and the two will be merged as : `{col_value} ({stat_value})` and will keep the name of the original column.
- If significance_stat is `None` all columns are left unchanged.

#### Example with p-values

```python
df = pd.DataFrame(
    {
        "Est_Coeff": [0.03, 0.41, -0.28],
        "Est_Coeff_pvalue": [0.041, 0.002, 0.018],
    },
    index=["Intercept", "GDP growth", "Inflation"],
)

table = build_table(
    df=df,
    style="academic",
    significance_stat="pvalue",
    significance_layout="stack",
)
```


Notes:

- stars are computed **only** from p-values
- if `significance_stat="stderr"` or `"tstat"`, the auxiliary statistic is shown in parentheses but no stars are added
- if `significance_stat=None`, no merging is performed

![Significance table](https://raw.githubusercontent.com/thibault-charbonnier/better-tables/master/static/significance-table.png)

---

### 5) MultiIndex support

MultiIndex handling is supported natively, without any additional parameters.

#### Row MultiIndex

A row `MultiIndex` is rendered as multiple stub columns on the left, with repeated labels displayed sparsely.


![Row MultiIndex table](https://raw.githubusercontent.com/thibault-charbonnier/better-tables/master/static/row-multiindex.png)

#### Column MultiIndex

A column `MultiIndex` is rendered as a grouped multi-level header.

![Column MultiIndex table](https://raw.githubusercontent.com/thibault-charbonnier/better-tables/master/static/column-multiindex.png)

---

### 6) LaTeX and HTML output

Two backends are supported through the same API :

- `to_latex()` for publication-grade LaTeX output
- `to_html()` for notebooks, dashboards or web integration

```python
latex = table.to_latex()
html = table.to_html()
```

#### Save as `.tex` or compile to PDF

Please note that compiling to PDF requires a LaTeX compiler as `pdflatex`, `xelatex` or `lualatex`.
Users must ensure that the specified engine is installed and available in the system PATH.

For example, MiKTeX can be installed from: https://miktex.org/download.

```python
table.to_latex(save_as_tex=True, path="outputs/my_table.tex")
table.to_latex(save_as_pdf=True, path="outputs/my_table.pdf")
```

---

### 7) Built-in styles

Three predefined styles are currently available, each designed for a slightly different reporting context.

- **`academic`**  
  The default style, designed for research notes, reports and publication-oriented tables. It offers a good trade-off between a light minimalist layout and a more explicit, highly readable table structure. This is the best default choice for regression tables, performance summaries and most professional documents.

- **`minimal`**  
  A stripped-down style with no outer border and no internal lines.  
  It is useful when you want a very light visual footprint, for example in dashboards or notebooks. 

- **`boxed`**  
  A more explicit, framed style with a full box, internal horizontal lines and inner vertical separators. It is helpful when readability through clear cell separation matters more than visual minimalism, for example in dense operational tables.


You can also override specific style fields without redefining an entire style:

```python
table = build_table(
    ...
    style_options={
        "bold_index": True,
        "highlight_bold": True,
        "highlight_color": "red",
    },
)
```

## API reference

### `build_table`

Main convenience constructor.

```python
build_table(
    df,
    *,
    title=None,
    caption=None,
    style="academic",
    title_align="center",
    caption_align="left",
    title_position="above",
    caption_position="below",
    formats=None,
    format_axis="rows",
    note=None,
    style_options=None,
    highlight_mode=None,
    highlight_axis="rows",
    default_format=None,
    significance_stat=None,
    significance_layout="stack",
    significance_thresholds=(0.10, 0.05, 0.01),
)
```

### `ReportTable`

Main rendering object returned by the `build_table` method.

#### Public methods

- `to_latex(save_as_tex=False, save_as_pdf=False, path=None, latex_engine=None, float_env=True)`
- `to_html()`

---


## License

MIT License

Copyright (c) 2026 better-tables Contributors

See `LICENSE` file for details.