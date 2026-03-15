from pathlib import Path
import os
import subprocess
from html import escape as _std_html_escape

from ..table.table_style import TableStyle

@staticmethod
def _escape_latex(text: str) -> str:
    """Basic LaTeX escaping for cell content and labels."""
    if not text:
        return ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for char, repl in replacements.items():
        out = out.replace(char, repl)
    return out

@staticmethod
def _escape_html(text: str) -> str:
    """Basic HTML escaping for cell content and labels."""
    if not text:
        return ""
    out = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return out

@staticmethod
def _build_latex_colspec(n_body_cols: int, style: TableStyle | None) -> str:
    """
    Build the LaTeX column specification string for \\begin{tabular}{...}.

    Parameters
    ----------
    n_body_cols : int
        Number of data columns (excluding the index stub).
    style : TableStyle or None
        Current style; controls outer and inner vertical rules.

    Returns
    -------
    str
        LaTeX column specification.
    """
    if style is None:
        return "l" * (n_body_cols + 1)

    s = style
    stub = "l"
    body_align_char = {"left": "l", "center": "c", "right": "r"}[s.body_align]

    cols = [stub] + [body_align_char] * n_body_cols

    if s.inner_vlines in ("inner", "all"):
        center = "|".join(cols)
    else:
        center = "".join(cols)

    if s.outer_border == "box":
        return "|" + center + "|"
    else:
        return center

@staticmethod
def _compile_latex_to_pdf(
    tex_path: str | Path,
    engine: str = "pdflatex"
) -> Path:
    """
    Compile a .tex file to PDF using a LaTeX engine.

    Parameters
    ----------
    tex_path : str or Path
        Path to the .tex file.

    engine : str, default 'pdflatex'
        LaTeX engine to use.

    Returns
    -------
    Path
        Path to the generated .pdf file.

    Raises
    ------
    RuntimeError
        If the engine is not found or compilation fails.
    """
    tex_path = Path(tex_path)
    workdir = tex_path.parent

    if engine is None or not Path(engine).exists():
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
        engine = f"{home}/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe"
        
    cmd = [
        engine,
        "-interaction=nonstopmode",
        tex_path.name,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=workdir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"LaTeX engine '{engine}' not found. "
            "Make sure it is installed and in your PATH."
        ) from e
    except subprocess.CalledProcessError as e:
        output = e.stdout.decode("utf-8", errors="ignore") if e.stdout else ""
        raise RuntimeError(
            f"LaTeX compilation failed (code {e.returncode}).\n"
            f"--- LaTeX output ---\n{output}"
        ) from e

    pdf_path = tex_path.with_suffix(".pdf")

    if not pdf_path.exists():
        raise RuntimeError(
            f"LaTeX compilation seems to have succeeded, but no PDF found at {pdf_path}."
        )

    for ext in (".aux", ".log", ".out"):
        aux = tex_path.with_suffix(ext)
        if aux.exists():
            aux.unlink()

    return pdf_path


def html_escape(text: str | None) -> str:
    """
    Escape a string for safe inclusion in HTML.
    """
    if text is None:
        return ""
    return _std_html_escape(str(text), quote=True)