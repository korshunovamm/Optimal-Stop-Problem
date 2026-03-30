#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный прогон экспериментов и генерация EXPERIMENTS_REPORT.md (таблицы + графики).
Запуск: conda-activate OptStopRandNN && cd repo && PYTHONPATH=. python 28-03-26/run_report_pipeline.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(PKG)]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backward_snapshots import fit_backward_snapshots
from build_model import (
    build_black_scholes,
    build_heston,
    build_market_model,
    build_payoff,
    clone_model_same_params,
)
from config import ExperimentParams
from hedge_simulation import max_hedge_losses
import risk_metrics as rm


def _price_row(
    name: str,
    algo: str,
    model_name: str,
    bundle,
    elapsed: float,
) -> Dict[str, Any]:
    return OrderedDict(
        [
            ("experiment", name),
            ("algo", algo),
            ("model", model_name),
            ("price", round(float(bundle.price_terminal_discounted), 6)),
            ("time_s", round(elapsed, 2)),
            ("path_gen_s", round(bundle.path_gen_seconds, 4)),
        ]
    )


def run_bs_price_suite(
    name: str,
    params: ExperimentParams,
    payoff_f,
    algos: Tuple[str, ...] = ("LSM", "NLSM", "RLSM"),
) -> List[Dict[str, Any]]:
    rows = []
    for algo in algos:
        model = build_market_model(params)
        t0 = time.time()
        bundle = fit_backward_snapshots(
            algo,
            model,
            payoff_f,
            hidden_size=params.hidden_size,
            nb_epochs_nlsm=params.nb_epochs_nlsm,
            rlsm_factors=params.rlsm_factors,
            train_eval_split=params.train_eval_split,
            seed=params.seed_train,
        )
        rows.append(_price_row(name, algo, params.stock_model, bundle, time.time() - t0))
    return rows


def run_hedge_suite(
    name: str,
    params: ExperimentParams,
    payoff_f,
    outdir: Path,
    algos: Tuple[str, ...] = ("LSM", "NLSM", "RLSM"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Tuple[str, str]]]:
    report_json: Dict[str, Any] = {}
    table: List[Dict[str, Any]] = []
    hist_paths = []

    for algo in algos:
        np.random.seed(params.seed_train)
        train_m = build_market_model(params)
        t0 = time.time()
        bundle = fit_backward_snapshots(
            algo,
            train_m,
            payoff_f,
            hidden_size=params.hidden_size,
            nb_epochs_nlsm=params.nb_epochs_nlsm,
            rlsm_factors=params.rlsm_factors,
            train_eval_split=params.train_eval_split,
            seed=params.seed_train,
        )
        fit_s = time.time() - t0

        np.random.seed(params.seed_hedge)
        test_m = clone_model_same_params(train_m, params.nb_paths_hedge_test)
        paths, _ = test_m.generate_paths()
        m_model, m_intr = max_hedge_losses(
            bundle, payoff_f, paths, params.hidden_size, params.bump_spot
        )
        st = rm.summary_stats(m_model)
        q = rm.var_cvar(m_model, 0.95)
        report_json[algo] = {
            "price": bundle.price_terminal_discounted,
            "fit_time_s": fit_s,
            "hedge_loss_model_summary": st,
            "var_cvar_95": q,
        }
        table.append(
            OrderedDict(
                [
                    ("suite", name),
                    ("algo", algo),
                    ("price", round(float(bundle.price_terminal_discounted), 6)),
                    ("fit_s", round(fit_s, 2)),
                    ("loss_mean", round(st["mean"], 6)),
                    ("loss_p95", round(st["p95"], 6)),
                    ("var95", round(q["var"], 6)),
                    ("cvar95", round(q["cvar"], 6)),
                    ("n_paths_hedge", params.nb_paths_hedge_test),
                ]
            )
        )

        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.hist(m_model, bins=45, density=True, alpha=0.78, color="#2a6f97", edgecolor="white", linewidth=0.3)
        ax.axvline(q["var"], color="#e76f51", linestyle="--", linewidth=1.2, label="VaR 95%")
        ax.set_title("Max диск. shortfall (V_model - \\Phi), " + algo)
        ax.set_xlabel("Потери")
        ax.set_ylabel("Плотность")
        ax.legend(fontsize=8)
        fig.tight_layout()
        hp = outdir / f"report_hedge_hist_{name}_{algo}.png"
        fig.savefig(hp, dpi=130)
        plt.close(fig)
        hist_paths.append((algo, hp.name))

    return table, report_json, hist_paths


def run_vol_sweep(
    base: ExperimentParams,
    payoff_f,
    vols: List[float],
    algos: Tuple[str, ...] = ("LSM", "NLSM"),
) -> List[Dict[str, Any]]:
    rows = []
    for vol in vols:
        for algo in algos:
            p = ExperimentParams(**{**base.__dict__, "volatility": vol})
            pf = build_payoff(p)
            model = build_black_scholes(p)
            bundle = fit_backward_snapshots(
                algo,
                model,
                pf,
                hidden_size=p.hidden_size,
                nb_epochs_nlsm=p.nb_epochs_nlsm,
                train_eval_split=p.train_eval_split,
                seed=p.seed_train + int(100 * vol),
            )
            np.random.seed(p.seed_hedge + int(vol * 100))
            test_m = clone_model_same_params(model, p.nb_paths_hedge_test)
            paths, _ = test_m.generate_paths()
            m_model, _ = max_hedge_losses(
                bundle, pf, paths, p.hidden_size, p.bump_spot
            )
            st = rm.summary_stats(m_model)
            q = rm.var_cvar(m_model, 0.95)
            rows.append(
                OrderedDict(
                    [
                        ("volatility", vol),
                        ("algo", algo),
                        ("price", round(float(bundle.price_terminal_discounted), 6)),
                        ("hedge_loss_mean", round(st["mean"], 6)),
                        ("cvar95", round(q["cvar"], 6)),
                    ]
                )
            )
    return rows


def _md_table(headers: List[str], rows: List[Dict[str, Any]], keys: List[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


def main():
    outdir = PKG / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    # При необходимости увеличьте nb_paths_train / nb_epochs_nlsm для более точных цен.
    bs_params = ExperimentParams(
        stock_model="BlackScholes",
        nb_paths_train=12000,
        nb_dates=36,
        hidden_size=28,
        nb_epochs_nlsm=22,
        nb_paths_hedge_test=2500,
        bump_spot=0.5,
        seed_train=42,
        seed_hedge=12345,
    )
    payoff_bs = build_payoff(bs_params)

    print("=== Black–Scholes: цены ===")
    bs_price_rows = run_bs_price_suite("bs_main", bs_params, payoff_bs)

    print("=== Black–Scholes: хедж ===")
    hedge_table, hedge_report, hedge_hists = run_hedge_suite(
        "bs_main", bs_params, payoff_bs, outdir
    )
    with open(outdir / "report_hedge_risk.json", "w", encoding="utf-8") as f:
        json.dump(hedge_report, f, indent=2, ensure_ascii=False)

    sweep_base = ExperimentParams(
        stock_model="BlackScholes",
        nb_paths_train=9000,
        nb_paths_hedge_test=1800,
        nb_dates=30,
        hidden_size=24,
        nb_epochs_nlsm=18,
        bump_spot=0.5,
    )
    vols = [0.2, 0.3, 0.4]
    print("=== Sweep по волатильности ===")
    sweep_rows = run_vol_sweep(sweep_base, payoff_bs, vols)

    print("=== Heston: цены (все три алгоритма) ===")
    heston_params = ExperimentParams(
        stock_model="Heston",
        nb_paths_train=8000,
        nb_dates=28,
        hidden_size=24,
        nb_epochs_nlsm=18,
    )
    payoff_h = build_payoff(heston_params)
    # build_market_model выберет Heston по строке
    heston_price_rows = run_bs_price_suite(
        "heston_main", heston_params, payoff_h
    )

    # --- графики сравнения цен ---
    algos_order = ["LSM", "NLSM", "RLSM"]
    prices_bs = [r["price"] for r in bs_price_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(algos_order))
    ax.bar(x, prices_bs, color=["#264653", "#2a9d8f", "#e9c46a"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(algos_order)
    ax.set_ylabel("Цена амер. пута")
    ax.set_title("Black–Scholes: сравнение алгоритмов (обучение)")
    fig.tight_layout()
    fig.savefig(outdir / "report_prices_bs_bar.png", dpi=130)
    plt.close(fig)

    prices_h = [r["price"] for r in heston_price_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x[: len(prices_h)], prices_h, color=["#264653", "#2a9d8f", "#e9c46a"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(algos_order)
    ax.set_ylabel("Цена амер. пута")
    ax.set_title("Heston: сравнение алгоритмов")
    fig.tight_layout()
    fig.savefig(outdir / "report_prices_heston_bar.png", dpi=130)
    plt.close(fig)

    # sweep plot CVaR
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for algo, color in [("LSM", "#264653"), ("NLSM", "#e76f51")]:
        xs = [r["volatility"] for r in sweep_rows if r["algo"] == algo]
        ys = [r["cvar95"] for r in sweep_rows if r["algo"] == algo]
        ax.plot(xs, ys, marker="o", label=algo, color=color, linewidth=2)
    ax.set_xlabel("Волатильность \\sigma")
    ax.set_ylabel("CVaR 95% (ошибка хеджа)")
    ax.set_title("Чувствительность риска хеджа к \\sigma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "report_sweep_cvar.png", dpi=130)
    plt.close(fig)

    # CSV dumps
    def write_csv(path: Path, rows: List[Dict[str, Any]]):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(outdir / "report_prices_bs.csv", bs_price_rows)
    write_csv(outdir / "report_prices_heston.csv", heston_price_rows)
    write_csv(outdir / "report_hedge_summary.csv", hedge_table)
    write_csv(outdir / "report_vol_sweep.csv", sweep_rows)

    # --- Markdown ---
    md_path = PKG / "EXPERIMENTS_REPORT.md"
    lines = []
    lines.append("# Отчёт по экспериментам: оптимальная остановка и хедж")
    lines.append("")
    lines.append("Сгенерировано скриптом `run_report_pipeline.py`. Все артефакты в каталоге `output/`.")
    lines.append("")
    lines.append("## 1. Цель и методология")
    lines.append("")
    lines.append(
        "- **Актив**: одномерный американский пут (`Put1Dim`), марковская модель цены (Black–Scholes или Heston).\n"
        "- **Алгоритмы**: LSM (полиномиальный базис), NLSM (нейросеть на шаге продолжения), RLSM (рандомизированная нейросеть + Ridge по последнему слою) — реализация в каталоге `28-03-26/` поверх пакета `optimal_stopping`.\n"
        "- **Обучение**: обратная индукция по одним и тем же параметрам сетки времени и числу траекторий (см. таблицы параметров ниже).\n"
        "- **Хедж**: дискретный самофинансируемый **дельта-хедж** по численной $\\Delta$ (центральные разности от оценки $V(S,t)$ из сохранённых снимков регрессии). Метрика — **максимум по шагам** дисконтированного shortfall $\\max_i e^{-r t_i}\\max(0, V^{\\mathrm{mod}}_i - \\Phi_i)$, где $\\Phi$ — стоимость реплицирующего портфеля long-option; по траекториям строится распределение, считаются квантили, **VaR** и **CVaR** (95%).\n"
        "- **Замечание**: для NLSM при очень малой ITM-выборке на шаге используется фолбэк (константа среднего продолжения); при достаточном `nb_paths_train` сети обучаются на всех релевантных шагах."
    )
    lines.append("")
    lines.append("## 2. Параметры основных прогонов")
    lines.append("")
    lines.append("### Black–Scholes (ценовой ряд и хедж)")
    lines.append("")
    lines.append(_md_table(
        ["Параметр", "Значение"],
        [
            {"p": "Spot $S_0$", "v": str(bs_params.spot)},
            {"p": "Страйк $K$", "v": str(bs_params.strike)},
            {"p": "Срок $T$", "v": str(bs_params.maturity)},
            {"p": "$r$ (drift модели)", "v": str(bs_params.drift)},
            {"p": "$\\sigma$", "v": str(bs_params.volatility)},
            {"p": "Дивиденд", "v": str(bs_params.dividend)},
            {"p": "Число шагов $N$", "v": str(bs_params.nb_dates)},
            {"p": "Путей обучения", "v": str(bs_params.nb_paths_train)},
            {"p": "Путей для хеджа (тест)", "v": str(bs_params.nb_paths_hedge_test)},
            {"p": "Train/eval split", "v": str(bs_params.train_eval_split)},
            {"p": "NLSM: hidden, epochs", "v": f"{bs_params.hidden_size}, {bs_params.nb_epochs_nlsm}"},
            {"p": "RLSM: hidden", "v": str(bs_params.hidden_size)},
            {"p": "Бамп для $\\Delta$", "v": str(bs_params.bump_spot)},
            {"p": "Сиды train / hedge", "v": f"{bs_params.seed_train}, {bs_params.seed_hedge}"},
        ],
        ["p", "v"],
    ))
    lines.append("")
    lines.append("### Heston (только цены)")
    lines.append("")
    lines.append(_md_table(
        ["Параметр", "Значение"],
        [
            {"p": "Путей, шагов", "v": f"{heston_params.nb_paths_train}, {heston_params.nb_dates}"},
            {"p": "$\\kappa, \\theta, \\rho$ (speed, mean, corr)", "v": f"{heston_params.heston_speed}, {heston_params.heston_mean}, {heston_params.heston_corr}"},
            {"p": "Vol/sigma Heston, hidden, NLSM epochs", "v": f"{heston_params.volatility}, {heston_params.hidden_size}, {heston_params.nb_epochs_nlsm}"},
        ],
        ["p", "v"],
    ))
    lines.append("")
    lines.append("### Чувствительность к $\\sigma$ (свип)")
    lines.append("")
    lines.append(_md_table(
        ["Параметр", "Значение"],
        [
            {"p": "Сетка $\\sigma$", "v": ", ".join(map(str, vols))},
            {"p": "Путей обучение / хедж", "v": f"{sweep_base.nb_paths_train}, {sweep_base.nb_paths_hedge_test}"},
            {"p": "$N$, hidden, NLSM epochs", "v": f"{sweep_base.nb_dates}, {sweep_base.hidden_size}, {sweep_base.nb_epochs_nlsm}"},
        ],
        ["p", "v"],
    ))
    lines.append("")

    lines.append("## 3. Сравнение цен (Black–Scholes)")
    lines.append("")
    lines.append(_md_table(
        ["Эксп.", "Алгоритм", "Модель", "Цена", "Время, с", "Ген. путей, с"],
        bs_price_rows,
        ["experiment", "algo", "model", "price", "time_s", "path_gen_s"],
    ))
    lines.append("")
    lines.append("![Цены BS](output/report_prices_bs_bar.png)")
    lines.append("")
    lines.append("## 4. Сравнение цен (Heston, 1D без дисперсии в регрессорах)")
    lines.append("")
    lines.append(_md_table(
        ["Эксп.", "Алгоритм", "Модель", "Цена", "Время, с", "Ген. путей, с"],
        heston_price_rows,
        ["experiment", "algo", "model", "price", "time_s", "path_gen_s"],
    ))
    lines.append("")
    lines.append("![Цены Heston](output/report_prices_heston_bar.png)")
    lines.append("")

    lines.append("## 5. Риск дискретного хеджа (Black–Scholes)")
    lines.append("")
    lines.append(_md_table(
        ["Серия", "Алгоритм", "Цена", "Обучение, с", "Mean loss", "p95", "VaR95", "CVaR95", "Путей тест"],
        hedge_table,
        ["suite", "algo", "price", "fit_s", "loss_mean", "loss_p95", "var95", "cvar95", "n_paths_hedge"],
    ))
    lines.append("")
    for algo, fname in hedge_hists:
        lines.append(f"### Распределение shortfall — {algo}")
        lines.append("")
        lines.append(f"![Hedge {algo}](output/{fname})")
        lines.append("")

    lines.append("## 6. Влияние волатильности на CVaR ошибки хеджа")
    lines.append("")
    lines.append(_md_table(
        ["$\\sigma$", "Алгоритм", "Цена", "Mean loss", "CVaR95"],
        sweep_rows,
        ["volatility", "algo", "price", "hedge_loss_mean", "cvar95"],
    ))
    lines.append("")
    lines.append("![Свип CVaR](output/report_sweep_cvar.png)")
    lines.append("")

    lines.append("## 7. Файлы данных (CSV/JSON)")
    lines.append("")
    lines.append(
        "- `output/report_prices_bs.csv`, `output/report_prices_heston.csv`\n"
        "- `output/report_hedge_summary.csv`, `output/report_hedge_risk.json`\n"
        "- `output/report_vol_sweep.csv`"
    )
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Done. Wrote", md_path)


if __name__ == "__main__":
    main()
