import numpy as np
import pandas as pd

from src.better_tables import build_table, MetricFormat


def test_basic_table():
    """
    Cas 1 :
    tableau simple, formats par colonnes, highlight max par colonne.
    Permet de valider :
    - default formatting
    - formats simples
    - highlight
    - rendu standard
    """
    df = pd.DataFrame(
        {
            "Return": [0.1245, 0.0832, -0.0211, 0.1567],
            "Volatility": [0.18, 0.12, 0.09, 0.21],
            "Sharpe": [0.69, 0.69, -0.23, 0.75],
        },
        index=["Strategy A", "Strategy B", "Strategy C", "Strategy D"],
    )

    formats = {
        "Return": {"kind": "percent", "percent_input": "ratio", "decimals": 1},
        "Volatility": {"kind": "percent", "percent_input": "ratio", "decimals": 1},
        "Sharpe": {"kind": "number", "decimals": 2},
    }

    table = build_table(
        df=df,
        title="Test 1 - Basic Performance Table",
        caption="Simple table with standard formatting and column-wise highlighting.",
        style="academic",
        formats=formats,
        format_axis="columns",
        highlight_mode="max",
        highlight_axis="columns",
        note="Used to validate the standard rendering pipeline.",
    )
    return table


def test_percent_logic_table():
    """
    Cas 2 :
    test explicite de la logique de pourcentage.
    Permet de valider :
    - kind='percent'
    - percent_input='ratio'
    - percent_input='percent'
    - coexistence de différents formats
    """
    df = pd.DataFrame(
        {
            "Return_ratio": [0.0345, 0.0712, -0.0125],
            "Vol_ratio": [0.152, 0.184, 0.111],
            "Already_percent": [3.45, 7.12, -1.25],
            "Plain_number": [1.2345, 2.3456, 3.4567],
        },
        index=["Portfolio A", "Portfolio B", "Portfolio C"],
    )

    formats = {
        "Return_ratio": {"kind": "percent", "percent_input": "ratio", "decimals": 2},
        "Vol_ratio": {"kind": "percent", "percent_input": "ratio", "decimals": 1},
        "Already_percent": {"kind": "percent", "percent_input": "percent", "decimals": 2},
        "Plain_number": {"kind": "number", "decimals": 3},
    }

    table = build_table(
        df=df,
        title="Test 2 - Percent Logic",
        caption="Comparison between ratio inputs and already-percent inputs.",
        style="academic",
        formats=formats,
        format_axis="columns",
        note="This table is mainly intended to validate percent rendering semantics.",
    )
    return table


def test_significance_flat_table():
    """
    Cas 3 :
    significativité sur colonnes plates.
    Permet de valider :
    - significance_stat='pvalue'
    - significance_layout='stack'
    - étoiles de significativité
    - suppression visuelle des colonnes _pvalue
    """
    index = [
        "Intercept",
        "GDP growth",
        "Inflation",
        "Unemployment rate",
        "Interest rate",
        "Credit spread",
    ]

    df = pd.DataFrame(
        {
            "Coefficient": [0.03, 0.41, -0.28, -0.18, -0.09, -0.52],
            "Coefficient_pvalue": [0.041, 0.002, 0.018, 0.094, 0.310, 0.001],
        },
        index=index,
    )

    formats = {
        "Coefficient": {"kind": "number", "decimals": 2},
    }

    table = build_table(
        df=df,
        title="Test 3 - Significance (Flat Columns)",
        caption="Regression output with p-values stacked below coefficients.",
        style="academic",
        formats=formats,
        format_axis="columns",
        significance_stat="pvalue",
        significance_layout="stack",
        significance_thresholds=(0.10, 0.05, 0.01),
        note="Stars are computed from p-values only.",
    )
    return table


def test_multiindex_index_table():
    """
    Cas 4 :
    MultiIndex en index.
    Permet de valider :
    - stub multi-colonnes à gauche
    - sparsification correcte
    - formats par colonnes
    - highlight sur colonnes
    """
    np.random.seed(42)

    stocks = ["AAPL", "MSFT", "GOOG", "NVDA"]
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    index = pd.MultiIndex.from_product([stocks, dates], names=["Stock", "Date"])

    n = len(index)
    df = pd.DataFrame(
        {
            "Return": np.random.normal(0.001, 0.01, size=n),
            "Volatility": np.random.normal(0.02, 0.004, size=n),
            "Volume": np.random.randint(1_000_000, 5_000_000, size=n),
        },
        index=index,
    )

    formats = {
        "Return": {"kind": "number", "decimals": 3},
        "Volatility": {"kind": "number", "decimals": 3},
        "Volume": {"kind": "number", "formatter": lambda v: f"{int(v):,}"},
    }

    table = build_table(
        df=df,
        title="Test 4 - MultiIndex on Index",
        caption="Synthetic stock data indexed by (Stock, Date).",
        style="academic",
        formats=formats,
        format_axis="columns",
        highlight_mode="max",
        highlight_axis="columns",
        note="Used to validate sparse rendering of MultiIndex row labels.",
    )
    return table


def test_multiindex_columns_significance_table():
    """
    Cas 5 :
    MultiIndex en colonnes + significativité.
    Permet de valider :
    - header multiniveau
    - regroupement visuel par modèle
    - logique significance sur dernier niveau
    - stack des p-values
    """
    index = [
        "Intercept",
        "GDP growth",
        "Inflation",
        "Unemployment rate",
        "Interest rate",
        "Credit spread",
        "VIX",
        "Oil price",
    ]

    columns = pd.MultiIndex.from_tuples(
        [
            ("Model 1", "M_A"),
            ("Model 1", "M_A_pvalue"),
            ("Model 1", "M_B"),
            ("Model 1", "M_B_pvalue"),
            ("Model 2", "M_A"),
            ("Model 2", "M_A_pvalue"),
            ("Model 2", "M_B"),
            ("Model 2", "M_B_pvalue"),
        ]
    )

    values = np.array(
        [
            [0.03, 0.041, 0.03, 0.065, 0.02, 0.110, 0.03, 0.080],
            [0.41, 0.002, 0.40, 0.003, 0.43, 0.001, 0.38, 0.005],
            [-0.28, 0.018, -0.26, 0.024, -0.29, 0.012, -0.24, 0.031],
            [-0.18, 0.094, -0.17, 0.110, -0.20, 0.078, -0.16, 0.130],
            [-0.09, 0.310, -0.09, 0.350, -0.10, 0.280, -0.08, 0.390],
            [-0.52, 0.001, -0.51, 0.001, -0.54, 0.001, -0.49, 0.002],
            [-0.01, 0.430, -0.01, 0.500, -0.01, 0.400, -0.01, 0.550],
            [0.05, 0.067, 0.05, 0.055, 0.04, 0.080, 0.06, 0.045],
        ]
    )

    df = pd.DataFrame(values, index=index, columns=columns)

    formats = {
        ("Model 1", "M_A"): {"kind": "number", "decimals": 2},
        ("Model 1", "M_B"): {"kind": "number", "decimals": 2},
        ("Model 2", "M_A"): {"kind": "number", "decimals": 2},
        ("Model 2", "M_B"): {"kind": "number", "decimals": 2},
    }

    table = build_table(
        df=df,
        title="Test 5 - MultiIndex Columns + Significance",
        caption="Regression outputs grouped by model with p-values stacked below coefficients.",
        style="academic",
        formats=formats,
        format_axis="columns",
        significance_stat="pvalue",
        significance_layout="stack",
        significance_thresholds=(0.10, 0.05, 0.01),
        note="Used to validate MultiIndex column rendering together with significance handling.",
    )
    return table
 
if __name__ == "__main__":
    # Publi
    test_basic_table().to_latex(save_as_pdf=True, path="test_basic_table.pdf")
    test_percent_logic_table().to_latex(save_as_pdf=True, path="test_percent_logic_table.pdf")
    test_significance_flat_table().to_latex(save_as_pdf=True, path="test_significance_flat_table.pdf")
    test_multiindex_index_table().to_latex(save_as_pdf=True, path="test_multiindex_index_table.pdf")
    test_multiindex_columns_significance_table().to_latex(save_as_pdf=True, path="test_multiindex_columns_significance_table.pdf")