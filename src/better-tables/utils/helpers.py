from typing import Sequence, Literal

from ..layout.layout_blocks import Row


def _normalize_row_heights(rows: Sequence[Row]) -> list[float]:
    """
    Normalize row height weights so that they sum to one.

    If all rows have ``height=None``, equal weights are assigned. If some
    rows have explicit heights and others are ``None``, the ``None`` rows
    are treated as having weight 1.0 before normalization.

    Parameters
    ----------
    rows : Sequence[Row]
        Rows whose ``height`` weights should be normalized.

    Returns
    -------
    list of float
        Normalized height weights, one per row. The list is empty if
        ``rows`` is empty.
    """
    if not rows:
        return []

    raw: list[float] = []
    for r in rows:
        if r.height is None:
            raw.append(1.0)
        else:
            raw.append(float(r.height))

    total = sum(raw)
    if total <= 0:
        return [1.0 / len(rows)] * len(rows)

    return [h / total for h in raw]


def _normalize_block_widths(row: Row) -> list[float]:
    """
    Normalize block width weights within a row so that they sum to one.

    If all blocks have ``width=None``, equal weights are assigned. If some
    blocks have explicit widths and others are ``None``, the ``None`` blocks
    are treated as having weight 1.0 before normalization.

    Parameters
    ----------
    row : Row
        Row whose block widths should be normalized.

    Returns
    -------
    list of float
        Normalized width weights, one per block. The list is empty if
        the row has no blocks.
    """
    if not row.blocks:
        return []

    raw: list[float] = []
    for b in row.blocks:
        if b.width is None:
            raw.append(1.0)
        else:
            raw.append(float(b.width))

    total = sum(raw)
    if total <= 0:
        return [1.0 / len(row.blocks)] * len(row.blocks)

    return [w / total for w in raw]

def _aligned_env(text: str, align: Literal["left", "center", "right"]) -> list[str]:
    """
    Wrap a single line of text in a LaTeX horizontal alignment environment.

    Parameters
    ----------
    text : str
        Text to wrap. It is assumed to be a LaTeX fragment already escaped
        as needed by the caller.
        
    align : {'left', 'center', 'right'}
        Desired horizontal alignment.

    Returns
    -------
    list of str
        Lines forming the LaTeX environment wrapping the text.
    """
    if align == "center":
        env = "center"
    elif align == "right":
        env = "flushright"
    else:
        env = "flushleft"

    return [rf"\begin{{{env}}}", text, rf"\end{{{env}}}"]