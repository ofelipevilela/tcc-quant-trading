# -*- coding: utf-8 -*-
"""
Gera material visual da rodada de metricas ANFIS para apresentacao.

O foco deste script e transformar os artefatos ja gerados pelos experimentos
em figuras autoexplicativas para discussao com a orientadora. Ele nao roda
novos backtests; apenas consolida os resultados existentes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

EXPERIMENTS = [
    {
        "label": "Iteracao 3\nATR h10",
        "short": "iter3_atr_h10",
        "path": OUTPUTS / "experiment_iter03_h10_20260416_112808",
        "alvo": "Barreira ATR",
        "rr": "n/d",
        "descricao": "20k barras, horizonte 10, target por barreira ATR",
    },
    {
        "label": "RR 1:1\n50k",
        "short": "rr1_50k",
        "path": OUTPUTS / "experiment_rr50k_full_rr1_20260416_163836",
        "alvo": "Risco-retorno",
        "rr": "1:1",
        "descricao": "50k barras, stop estrutural, alvo 1R",
    },
    {
        "label": "RR 1:2\n50k",
        "short": "rr2_50k",
        "path": OUTPUTS / "experiment_rr50k_full_rr2_20260416_164258",
        "alvo": "Risco-retorno",
        "rr": "1:2",
        "descricao": "50k barras, stop estrutural, alvo 2R",
    },
]

BEST_SHORT = "rr2_50k"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#f7f8fa",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d0d5dd",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

COLORS = {
    "iter3_atr_h10": "#4e79a7",
    "rr1_50k": "#59a14f",
    "rr2_50k": "#e15759",
}


def pct(value: float) -> float:
    return float(value) * 100.0


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def compute_payoff_from_trades(exp_dir: Path) -> float:
    ratios: list[float] = []
    for trade_file in sorted(exp_dir.glob("fold_*_trades.csv")):
        trades = pd.read_csv(trade_file)
        if "return_pct" not in trades.columns or trades.empty:
            continue
        returns = pd.to_numeric(trades["return_pct"], errors="coerce")
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if wins.empty or losses.empty:
            continue
        ratios.append(float(wins.mean() / abs(losses.mean())))
    return float(np.mean(ratios)) if ratios else float("nan")


def value_from(exp: dict[str, Any], aggregate_key: str, fold_col: str | None = None) -> float:
    aggregate = exp["aggregate"]
    folds = exp["fold_metrics"]
    if aggregate_key in aggregate:
        return float(aggregate[aggregate_key])
    if fold_col and fold_col in folds.columns:
        return safe_mean(folds[fold_col])
    return float("nan")


def load_experiments() -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        config_path = spec["path"] / "config_and_results.json"
        metrics_path = spec["path"] / "fold_metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Artefatos nao encontrados em {spec['path']}")

        payload = read_json(config_path)
        fold_metrics = pd.read_csv(metrics_path)
        aggregate = payload.get("aggregate", {})
        config = payload.get("config", {})
        loaded.append(
            {
                **spec,
                "payload": payload,
                "config": config,
                "aggregate": aggregate,
                "fold_metrics": fold_metrics,
                "computed_payoff": compute_payoff_from_trades(spec["path"]),
            }
        )
    return loaded


def build_summary(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        payoff = value_from(exp, "mean_test_payoff_ratio", "test_payoff_ratio")
        if np.isnan(payoff):
            payoff = exp["computed_payoff"]

        rows.append(
            {
                "configuracao": exp["label"].replace("\n", " "),
                "short": exp["short"],
                "descricao": exp["descricao"],
                "barras": exp["config"].get("data", {}).get("n_bars", np.nan),
                "target": exp["config"].get("target", {}).get("mode", "atr_barrier"),
                "horizon": exp["config"].get("features", {}).get("horizon", np.nan),
                "rr": exp["config"].get("target", {}).get("reward_to_risk", exp["rr"]),
                "stop_mode": exp["config"].get("target", {}).get(
                    "stop_mode", exp["config"].get("backtest", {}).get("stop_mode", "atr")
                ),
                "min_activation": exp["config"].get("features", {}).get("min_activation", np.nan),
                "folds": value_from(exp, "folds"),
                "mae": value_from(exp, "mean_test_mae", "test_mae"),
                "rmse": value_from(exp, "mean_test_rmse", "test_rmse"),
                "r2": value_from(exp, "mean_test_r2", "test_r2"),
                "da_bruta_pct": pct(value_from(exp, "mean_test_directional_accuracy", "test_directional_accuracy")),
                "da_filtrada_pct": pct(
                    value_from(
                        exp,
                        "mean_thresholded_test_directional_accuracy",
                        "thresholded_test_directional_accuracy",
                    )
                ),
                "coverage_pct": pct(
                    value_from(exp, "mean_thresholded_test_coverage", "thresholded_test_coverage")
                ),
                "pf": value_from(exp, "mean_test_profit_factor", "test_profit_factor"),
                "win_rate_pct": value_from(exp, "mean_test_win_rate", "test_win_rate"),
                "payoff": payoff,
                "retorno_pct": value_from(exp, "mean_test_total_return", "test_total_return"),
                "retorno_anualizado_pct": value_from(
                    exp, "mean_test_annualized_return", "test_annualized_return"
                ),
                "max_dd_pct": value_from(exp, "mean_test_max_drawdown", "test_max_drawdown"),
                "trades": value_from(exp, "mean_test_total_trades", "test_total_trades"),
                "pf_gt_1": value_from(exp, "positive_pf_folds"),
                "retorno_pos_folds": value_from(exp, "positive_return_folds"),
                "annual_ge_12_folds": value_from(exp, "annual_return_ge_12_folds"),
                "rule_dominance": value_from(exp, "mean_test_rule_dominance", "test_rule_dominance"),
                "active_rules_95": value_from(exp, "mean_test_active_rules_95", "test_active_rules_95"),
                "mf_center_shift": value_from(exp, "mean_mf_mean_abs_center_shift"),
                "mf_sigma_shift": value_from(exp, "mean_mf_mean_abs_sigma_shift"),
                "score_target_corr": value_from(exp, "mean_test_score_target_corr", "test_score_target_corr"),
                "sharpe": value_from(exp, "mean_test_sharpe_ratio", "test_sharpe_ratio"),
                "sortino": value_from(exp, "mean_test_sortino_ratio", "test_sortino_ratio"),
                "sqn": value_from(exp, "mean_test_sqn", "test_sqn"),
            }
        )
    return pd.DataFrame(rows)


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", suffix: str = "") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height) + suffix,
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_resumo_executivo(summary: pd.DataFrame, out_dir: Path) -> None:
    best = summary.loc[summary["short"] == BEST_SHORT].iloc[0]
    cards = [
        ("Profit Factor", f"{best['pf']:.3f}", "Acima de 1 indica lucro bruto maior que perda bruta."),
        ("Retorno médio", f"{best['retorno_pct']:.2f}%", "Media dos retornos nos folds de teste."),
        ("DA filtrada", f"{best['da_filtrada_pct']:.2f}%", "Acuracia direcional apenas quando o modelo operou."),
        ("Payoff", f"{best['payoff']:.2f}", "Ganho medio por trade vencedor dividido pela perda media."),
        ("Folds PF > 1", f"{int(best['pf_gt_1'])}/4", "Consistencia operacional entre janelas temporais."),
        ("R² medio", f"{best['r2']:.4f}", "Baixo, como esperado em serie financeira ruidosa."),
    ]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_axis_off()
    ax.set_title(
        "Resumo executivo da melhor rodada: ANFIS causal_v3 com target RR 1:2",
        fontsize=18,
        pad=20,
    )
    ax.text(
        0.5,
        0.88,
        "O teste com 50.000 barras e alvo de 2R foi o melhor recorte observado: "
        "menor win rate, mas maior payoff e PF positivo em todos os folds.",
        ha="center",
        va="center",
        fontsize=12,
        color="#344054",
        transform=ax.transAxes,
    )

    positions = [(0.05, 0.55), (0.37, 0.55), (0.69, 0.55), (0.05, 0.22), (0.37, 0.22), (0.69, 0.22)]
    for (title, value, note), (x, y) in zip(cards, positions):
        ax.add_patch(
            Rectangle(
                (x, y),
                0.26,
                0.22,
                transform=ax.transAxes,
                facecolor="#ffffff",
                edgecolor="#d0d5dd",
                linewidth=1.2,
            )
        )
        ax.text(x + 0.13, y + 0.15, value, ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(x + 0.13, y + 0.10, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + 0.13, y + 0.045, note, ha="center", va="center", fontsize=8.5, color="#475467", wrap=True)

    ax.text(
        0.5,
        0.08,
        "Nota metodologica: o retorno anualizado superou 12% nos 4 folds, mas deve ser tratado "
        "como indicador interno do backtest, nao como promessa de desempenho real.",
        ha="center",
        fontsize=10,
        color="#667085",
        transform=ax.transAxes,
    )
    save_current_figure(out_dir / "00_resumo_executivo_rr2.png")


def plot_parametros(summary: pd.DataFrame, out_dir: Path) -> None:
    columns = [
        "configuracao",
        "barras",
        "target",
        "horizon",
        "rr",
        "stop_mode",
        "min_activation",
        "folds",
    ]
    table_df = summary[columns].copy()
    table_df.columns = [
        "Configuracao",
        "Barras",
        "Alvo",
        "Horizonte",
        "RR",
        "Stop",
        "Min. ativacao",
        "Folds",
    ]

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.set_axis_off()
    ax.set_title("Parametros comparados nos experimentos", fontsize=16, pad=20)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
            cell.set_facecolor("#344054")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f2f4f7")
    ax.text(
        0.5,
        0.05,
        "A mudanca central foi sair do alvo temporal por barreira ATR para um alvo em R, "
        "com stop estrutural e preservacao da linha temporal completa.",
        ha="center",
        color="#475467",
        transform=ax.transAxes,
    )
    save_current_figure(out_dir / "01_parametros_experimentos.png")


def plot_metricas_academicas(summary: pd.DataFrame, out_dir: Path) -> None:
    labels = summary["configuracao"].tolist()
    colors = [COLORS[s] for s in summary["short"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Metricas academicas de regressao e direcao", fontsize=16, fontweight="bold")

    plots = [
        ("mae", "MAE no teste", "Menor e melhor", "{:.3f}", ""),
        ("rmse", "RMSE no teste", "Menor e melhor; penaliza erros grandes", "{:.3f}", ""),
        ("r2", "R² no teste", "Proximo de zero e comum em series financeiras curtas", "{:.4f}", ""),
    ]
    for ax, (col, title, subtitle, fmt, suffix) in zip(axes.flat[:3], plots):
        ax.bar(x, summary[col], color=colors)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.set_ylabel(col.upper())
        ax.text(0.02, 0.92, subtitle, transform=ax.transAxes, fontsize=9, color="#667085")
        if col == "r2":
            ax.axhline(0, color="#101828", linewidth=1)
        annotate_bars(ax, fmt=fmt, suffix=suffix)

    ax = axes.flat[3]
    width = 0.35
    ax.bar(x - width / 2, summary["da_bruta_pct"], width, label="DA bruta", color="#84c5f4")
    ax.bar(x + width / 2, summary["da_filtrada_pct"], width, label="DA filtrada", color="#f4a261")
    ax.axhline(50, color="#101828", linewidth=1, linestyle="--", label="Referencia 50%")
    ax.set_title("Acuracia direcional")
    ax.set_ylabel("%")
    ax.set_xticks(x, labels)
    ax.legend()
    annotate_bars(ax, fmt="{:.1f}", suffix="%")
    save_current_figure(out_dir / "02_metricas_academicas.png")


def plot_metricas_operacionais(summary: pd.DataFrame, out_dir: Path) -> None:
    labels = summary["configuracao"].tolist()
    colors = [COLORS[s] for s in summary["short"]]
    x = np.arange(len(labels))

    metrics = [
        ("pf", "Profit Factor", "Acima de 1 e desejavel", "{:.3f}", "", 1.0),
        ("win_rate_pct", "Win Rate", "Percentual de trades vencedores", "{:.1f}", "%", 50.0),
        ("payoff", "Payoff medio", "Ganho medio / perda media", "{:.2f}", "", 1.0),
        ("retorno_pct", "Retorno medio por fold", "Resultado percentual do capital", "{:.1f}", "%", 0.0),
        ("max_dd_pct", "Max Drawdown medio", "Menor e melhor", "{:.1f}", "%", None),
        ("coverage_pct", "Coverage operacional", "Percentual de amostras ativadas", "{:.1f}", "%", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Metricas operacionais comparativas", fontsize=16, fontweight="bold")

    for ax, (col, title, subtitle, fmt, suffix, ref) in zip(axes.flat, metrics):
        ax.bar(x, summary[col], color=colors)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.text(0.02, 0.92, subtitle, transform=ax.transAxes, fontsize=9, color="#667085")
        if ref is not None:
            ax.axhline(ref, color="#101828", linewidth=1, linestyle="--")
        annotate_bars(ax, fmt=fmt, suffix=suffix)

    save_current_figure(out_dir / "03_metricas_operacionais.png")


def plot_interpretabilidade(summary: pd.DataFrame, out_dir: Path) -> None:
    labels = summary["configuracao"].tolist()
    colors = [COLORS[s] for s in summary["short"]]
    x = np.arange(len(labels))
    metrics = [
        ("rule_dominance", "Dominancia de regras", "Quanto maior, mais concentrado em poucas regras.", "{:.3f}"),
        ("active_rules_95", "Regras ativas (>5%)", "Quantidade media de regras realmente usadas.", "{:.1f}"),
        ("mf_center_shift", "Deslocamento medio dos centros", "Mostra adaptacao das MFs apos treino.", "{:.3f}"),
        ("mf_sigma_shift", "Deslocamento medio dos sigmas", "Mostra mudanca na largura das MFs.", "{:.3f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Metricas de interpretabilidade neuro-fuzzy", fontsize=16, fontweight="bold")
    for ax, (col, title, subtitle, fmt) in zip(axes.flat, metrics):
        ax.bar(x, summary[col], color=colors)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.text(0.02, 0.92, subtitle, transform=ax.transAxes, fontsize=9, color="#667085")
        annotate_bars(ax, fmt=fmt)

    save_current_figure(out_dir / "04_interpretabilidade_regras_mfs.png")


def get_best_experiment(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    return next(exp for exp in experiments if exp["short"] == BEST_SHORT)


def plot_rr2_por_fold(best_exp: dict[str, Any], out_dir: Path) -> None:
    df = best_exp["fold_metrics"].copy()
    fold_labels = [f"Fold {int(v)}" for v in df["fold"]]
    x = np.arange(len(df))

    panels = [
        ("test_profit_factor", "Profit Factor por fold", "{:.3f}", "", 1.0),
        ("test_total_return", "Retorno total por fold", "{:.1f}", "%", 0.0),
        ("test_max_drawdown", "Max Drawdown por fold", "{:.1f}", "%", None),
        ("thresholded_test_directional_accuracy", "DA filtrada por fold", "{:.1f}", "%", 0.5),
        ("thresholded_test_coverage", "Coverage por fold", "{:.1f}", "%", None),
        ("test_total_trades", "Trades por fold", "{:.0f}", "", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Consistencia da melhor configuracao: RR 1:2 por fold", fontsize=16, fontweight="bold")
    for ax, (col, title, fmt, suffix, ref) in zip(axes.flat, panels):
        values = pd.to_numeric(df[col], errors="coerce")
        if col in {"thresholded_test_directional_accuracy", "thresholded_test_coverage"}:
            values = values * 100.0
            ref = 50.0 if ref == 0.5 else ref
        ax.bar(x, values, color="#e15759")
        ax.set_title(title)
        ax.set_xticks(x, fold_labels)
        if ref is not None:
            ax.axhline(ref, color="#101828", linewidth=1, linestyle="--")
        annotate_bars(ax, fmt=fmt, suffix=suffix)

    save_current_figure(out_dir / "05_rr2_metricas_por_fold.png")


def plot_meta_anualizada(best_exp: dict[str, Any], out_dir: Path) -> None:
    df = best_exp["fold_metrics"].copy()
    values = pd.to_numeric(df["test_annualized_return"], errors="coerce")
    labels = [f"Fold {int(v)}" for v in df["fold"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(labels, values, color="#e15759")
    ax.axhline(12.0, color="#101828", linewidth=1.5, linestyle="--", label="Meta de 12% a.a.")
    ax.set_yscale("log")
    ax.set_ylabel("Retorno anualizado (%) em escala log")
    ax.set_title("Comparacao com meta de 12% ao ano (indicador interno do backtest)")
    ax.legend()
    for i, value in enumerate(values):
        ax.annotate(f"{value:.0f}%", (i, value), ha="center", va="bottom", fontsize=9)
    ax.text(
        0.5,
        -0.16,
        "Leitura cautelosa: a anualizacao usa janelas curtas e composicao de capital; "
        "por isso serve como comparacao interna, nao como promessa operacional.",
        ha="center",
        transform=ax.transAxes,
        color="#667085",
    )
    save_current_figure(out_dir / "06_meta_12_por_cento_anualizada.png")


def plot_equity_curves(best_exp: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for fold in range(1, 5):
        path = best_exp["path"] / f"fold_{fold:02d}_equity.csv"
        equity = pd.read_csv(path)["equity"].astype(float)
        normalized_return = (equity / equity.iloc[0] - 1.0) * 100.0
        ax.plot(normalized_return.values, linewidth=2, label=f"Fold {fold}")

    ax.axhline(0, color="#101828", linewidth=1)
    ax.set_title("Curvas de capital por fold - RR 1:2")
    ax.set_xlabel("Passos do backtest no fold")
    ax.set_ylabel("Retorno acumulado (%)")
    ax.legend()
    ax.text(
        0.02,
        0.95,
        "Todas as curvas terminaram positivas, mas com drawdowns intermediarios.",
        transform=ax.transAxes,
        color="#667085",
    )
    save_current_figure(out_dir / "07_curvas_capital_rr2.png")


def plot_collage(
    best_exp: dict[str, Any],
    out_dir: Path,
    pattern: str,
    title: str,
    subtitle: str,
    filename: str,
) -> None:
    files = [best_exp["path"] / pattern.format(fold=fold) for fold in range(1, 5)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for idx, (ax, path) in enumerate(zip(axes.flat, files), start=1):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(f"Fold {idx}")
        ax.set_axis_off()
    fig.text(0.5, 0.03, subtitle, ha="center", color="#667085")
    save_current_figure(out_dir / filename)


def plot_target_rr_scheme(out_dir: Path) -> None:
    x = np.arange(14)
    price = np.array([100, 99.6, 99.1, 98.6, 99.2, 100.0, 100.7, 101.1, 100.8, 101.8, 102.7, 103.3, 103.0, 104.0])
    entry_idx = 5
    entry = price[entry_idx]
    stop = 98.4
    risk = entry - stop
    tp1 = entry + risk
    tp2 = entry + 2 * risk

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, price, marker="o", linewidth=2.5, color="#344054")
    ax.scatter([3], [price[3]], s=120, color="#4e79a7", label="Sweep / fundo estrutural")
    ax.scatter([entry_idx], [entry], s=120, color="#59a14f", label="Entrada apos realinhamento")
    ax.axhline(stop, color="#d92d20", linestyle="--", linewidth=2, label="Stop estrutural")
    ax.axhline(tp1, color="#f79009", linestyle="--", linewidth=2, label="Alvo 1R")
    ax.axhline(tp2, color="#12b76a", linestyle="--", linewidth=2, label="Alvo 2R")
    ax.fill_between(x, stop, entry, color="#fecdca", alpha=0.35, label="Risco")
    ax.fill_between(x, entry, tp2, color="#d1fadf", alpha=0.25, label="Potencial 2R")
    ax.set_title("Esquema conceitual do novo target por risco-retorno")
    ax.set_xlabel("Candles futuros dentro do horizonte maximo")
    ax.set_ylabel("Preco ilustrativo")
    ax.legend(loc="upper left")
    ax.text(
        0.55,
        0.08,
        "O alvo RR troca a pergunta 'onde o preco estara daqui a N candles?' por "
        "'qual direcao teria melhor resultado em R antes do tempo limite?'.",
        transform=ax.transAxes,
        fontsize=10,
        color="#475467",
        bbox=dict(facecolor="#ffffff", edgecolor="#d0d5dd", boxstyle="round,pad=0.5"),
    )
    save_current_figure(out_dir / "10_esquema_target_rr.png")


def write_readme(summary: pd.DataFrame, best_exp: dict[str, Any], out_dir: Path) -> None:
    best = summary.loc[summary["short"] == BEST_SHORT].iloc[0]
    text = f"""# Material visual para orientadora - metricas ANFIS

Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

## Resultado principal

A melhor configuracao desta rodada foi o ANFIS `causal_v3` com 50.000 barras,
target por risco-retorno e RR 1:2. O resultado medio nos 4 folds foi:

- Profit Factor: {best['pf']:.3f}
- Retorno medio por fold: {best['retorno_pct']:.2f}%
- DA filtrada: {best['da_filtrada_pct']:.2f}%
- Coverage: {best['coverage_pct']:.2f}%
- Win Rate: {best['win_rate_pct']:.2f}%
- Payoff: {best['payoff']:.2f}
- Max Drawdown: {best['max_dd_pct']:.2f}%
- R2: {best['r2']:.4f}
- Folds com PF > 1: {int(best['pf_gt_1'])}/4

Interpretacao curta: o RR 1:2 nao venceu por ter o maior win rate, mas por
combinar payoff proximo de 2R com Profit Factor positivo nos quatro folds.
Isso sugere que o modelo esta mais adequado para filtrar deslocamentos com
assimetria do que para buscar alta taxa de acerto em alvo curto.

## Imagens geradas

1. `00_resumo_executivo_rr2.png`
   - Mostra os principais indicadores da melhor configuracao.
   - Serve como slide de abertura da discussao.

2. `01_parametros_experimentos.png`
   - Compara os parametros das configuracoes testadas: quantidade de barras,
     tipo de target, horizonte, RR, stop e numero de folds.
   - A mudanca mais importante foi sair do alvo temporal por barreira ATR para
     alvo em R com stop estrutural.

3. `02_metricas_academicas.png`
   - Compara MAE, RMSE, R2 e acuracia direcional.
   - MAE e RMSE medem erro de regressao; quanto menores, melhor.
   - R2 proximo de zero e comum em series financeiras de curto prazo e deve ser
     interpretado junto com metricas operacionais.
   - DA filtrada mede a acuracia direcional apenas quando o modelo gerou sinal.

4. `03_metricas_operacionais.png`
   - Compara Profit Factor, Win Rate, Payoff, retorno, drawdown e coverage.
   - O RR 1:2 teve Win Rate menor, mas payoff e Profit Factor melhores.

5. `04_interpretabilidade_regras_mfs.png`
   - Mostra dominancia de regras, quantidade de regras ativas e deslocamento das
     funcoes de pertinencia.
   - Ajuda a explicar que o modelo neuro-fuzzy nao e apenas caixa-preta: tambem
     e possivel observar adaptacao das regras e das MFs.

6. `05_rr2_metricas_por_fold.png`
   - Detalha a melhor configuracao fold a fold.
   - O ponto mais relevante e que o Profit Factor ficou acima de 1 nos quatro
     folds, indicando consistencia dentro da janela testada.

7. `06_meta_12_por_cento_anualizada.png`
   - Compara cada fold com a meta de 12% ao ano.
   - A leitura deve ser cautelosa, pois a anualizacao usa janelas curtas e
     composicao de capital.

8. `07_curvas_capital_rr2.png`
   - Mostra as curvas de capital por fold.
   - Ajuda a visualizar retorno e drawdown, nao apenas metricas finais.

9. `08_curvas_aprendizado_rr2.png`
   - Agrupa as curvas de aprendizado exportadas pelo treinamento.
   - Serve para discutir estabilidade, reducao de erro e possiveis sinais de
     overfitting.

10. `09_funcoes_pertinencia_rr2.png`
    - Agrupa os graficos de evolucao das funcoes de pertinencia por fold.
    - E util para mostrar visualmente que o ANFIS ajustou seus conjuntos fuzzy.

11. `10_esquema_target_rr.png`
    - Figura conceitual do novo target por risco-retorno.
    - Explica a diferenca entre alvo por tempo e alvo por preco/risco.

## Explicacao das metricas

- MAE: erro absoluto medio. Menor e melhor.
- RMSE: erro quadratico medio com raiz. Penaliza erros grandes.
- R2: proporcao da variancia explicada pelo modelo. Em mercado financeiro
  intraday, valores proximos de zero ainda podem coexistir com algum valor
  operacional quando ha vantagem direcional ou assimetria.
- DA bruta: percentual de vezes em que o sinal previsto acertou o sinal real.
- DA filtrada: DA considerando apenas sinais acima do threshold operacional.
- Coverage: percentual de amostras em que o modelo gerou sinal operacional.
- Profit Factor: lucro bruto dividido pela perda bruta. Acima de 1 indica
  resultado positivo no recorte avaliado.
- Win Rate: percentual de trades vencedores.
- Payoff: ganho medio dos trades vencedores dividido pela perda media dos
  trades perdedores.
- Retorno total: variacao percentual do capital no fold.
- Max Drawdown: maior queda percentual da curva de capital.
- Rule dominance: concentracao media de ativacao em poucas regras fuzzy.
- Active rules 95: quantidade media de regras com uso relevante.
- MF shift: deslocamento medio dos centros/sigmas das funcoes de pertinencia.

## Cautela metodologica

Os resultados sao promissores dentro da bateria executada, mas ainda nao devem
ser apresentados como comprovacao definitiva de lucratividade real. A proxima
etapa recomendada e incluir custos operacionais, slippage, spread variavel e uma
janela final congelada de out-of-sample.

Artefatos usados como melhor configuracao:
`{best_exp['path'].as_posix()}`
"""
    (out_dir / "README_EXPLICACAO_METRICAS.md").write_text(text, encoding="utf-8")
    print(f"[OK] {out_dir / 'README_EXPLICACAO_METRICAS.md'}")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUTS / f"material_orientadora_metricas_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments()
    summary = build_summary(experiments)
    best_exp = get_best_experiment(experiments)

    summary.to_csv(out_dir / "metricas_comparativas.csv", index=False, encoding="utf-8")
    best_exp["fold_metrics"].to_csv(out_dir / "metricas_rr2_por_fold.csv", index=False, encoding="utf-8")

    plot_resumo_executivo(summary, out_dir)
    plot_parametros(summary, out_dir)
    plot_metricas_academicas(summary, out_dir)
    plot_metricas_operacionais(summary, out_dir)
    plot_interpretabilidade(summary, out_dir)
    plot_rr2_por_fold(best_exp, out_dir)
    plot_meta_anualizada(best_exp, out_dir)
    plot_equity_curves(best_exp, out_dir)
    plot_collage(
        best_exp,
        out_dir,
        "fold_{fold:02d}_learning_curve.png",
        "Curvas de aprendizado por fold - RR 1:2",
        "Comparacao visual entre treinamento e validacao para discutir estabilidade e overfitting.",
        "08_curvas_aprendizado_rr2.png",
    )
    plot_collage(
        best_exp,
        out_dir,
        "fold_{fold:02d}_mf_evolution.png",
        "Evolucao das funcoes de pertinencia por fold - RR 1:2",
        "As figuras mostram como centros e larguras das MFs foram ajustados apos o treinamento.",
        "09_funcoes_pertinencia_rr2.png",
    )
    plot_target_rr_scheme(out_dir)
    write_readme(summary, best_exp, out_dir)

    print(f"\nMaterial gerado em: {out_dir}")


if __name__ == "__main__":
    main()
