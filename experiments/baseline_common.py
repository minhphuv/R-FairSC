"""Shared experiment loop for sFairSC and standard SC."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from utils.metrics import compute_balance, normalized_cut


def _sample_std(values: np.ndarray) -> float:
    return float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_results(results: list[dict]) -> pd.DataFrame:
    rows = []

    for k in sorted({int(result["k"]) for result in results}):
        runs = [result for result in results if int(result["k"]) == k]
        balances = np.asarray([run["balance"] for run in runs], dtype=float)
        ncuts = np.asarray([run["ncut"] for run in runs], dtype=float)
        runtimes = np.asarray([run["runtime"] for run in runs], dtype=float)

        rows.append(
            {
                "k": k,
                "balance_mean": np.nanmean(balances),
                "balance_std": _sample_std(balances),
                "ncut_mean": np.nanmean(ncuts),
                "ncut_std": _sample_std(ncuts),
                "runtime_mean": np.nanmean(runtimes),
                "runtime_std": _sample_std(runtimes),
                "num_runs": len(runs),
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def run_baseline(
    *,
    method_name: str,
    solver,
    dataset,
    clusters: list[int],
    num_runs: int,
    output: str,
) -> pd.DataFrame:
    results: list[dict] = []

    for k in clusters:
        for run_index in range(num_runs):
            print(f"\n==== {method_name}: k={k} (run {run_index + 1}/{num_runs}) ====")

            start = time.perf_counter()
            solution = solver.fit_predict(k=k)
            runtime = time.perf_counter() - start

            balance = compute_balance(
                cluster_labels=solution.labels,
                group_indicator=dataset.group_indicator,
                k=k,
                verbose=False,
            )
            ncut = normalized_cut(dataset.adjacency, solution.labels)

            results.append(
                {
                    "k": k,
                    "balance": balance,
                    "ncut": ncut,
                    "runtime": runtime,
                }
            )

            print(
                f"[{method_name}, k={k}, run={run_index + 1}/{num_runs}] "
                f"balance={balance:.4f} ncut={ncut:.6f} "
                f"runtime={runtime:.3f}s bins={solution.cluster_sizes}"
            )

    summary = summarize_results(results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"\nWrote summary to {output_path}")
    return summary
