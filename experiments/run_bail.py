"""Run R-FairSC on the Bail dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from datasets.bail import load_bail_dataset
from solvers.r_fairsc import RFairSC, RFairSCConfig
from utils.metrics import compute_balance, normalized_cut


def _sample_std(values: np.ndarray) -> float:
    return float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_results(results: list[dict]) -> pd.DataFrame:
    """Aggregate repeated runs for each number of clusters."""
    rows = []

    for k in sorted({int(result["k"]) for result in results}):
        runs = [result for result in results if int(result["k"]) == k]

        balances = np.asarray([run["balance"] for run in runs], dtype=float)
        ncuts = np.asarray([run["ncut"] for run in runs], dtype=float)
        runtimes = np.asarray([run["runtime"] for run in runs], dtype=float)
        h_runtimes = np.asarray([run["H_runtime"] for run in runs], dtype=float)
        fair_x = np.asarray([run["fair_x"] for run in runs], dtype=float)

        rows.append(
            {
                "k": k,
                "balance_mean": np.nanmean(balances),
                "balance_std": _sample_std(balances),
                "ncut_mean": np.nanmean(ncuts),
                "ncut_std": _sample_std(ncuts),
                "runtime_mean": np.nanmean(runtimes),
                "runtime_std": _sample_std(runtimes),
                "H_runtime_mean": np.nanmean(h_runtimes),
                "H_runtime_std": _sample_std(h_runtimes),
                "fair_x_mean": np.nanmean(fair_x),
                "fair_x_std": _sample_std(fair_x),
                "num_runs": len(runs),
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    dataset = load_bail_dataset(
        data_path=args.data,
        edges_path=args.edges,
        sensitive_attribute=args.sensitive_attribute,
    )

    for group_value, count in zip(dataset.group_values, dataset.group_counts):
        print(f"size of {args.sensitive_attribute} group {group_value}: {count}")

    solver_config = RFairSCConfig(
        alpha0=args.alpha0,
        max_admm_iterations=args.max_admm_iterations,
        max_rcg_iterations=args.max_rcg_iterations,
    )
    solver = RFairSC(
        adjacency=dataset.adjacency,
        degree_inv_sqrt=dataset.degree_inv_sqrt,
        fairness_matrix=dataset.fairness_matrix,
        config=solver_config,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"R-FairSC device: {solver.device}")

    results: list[dict] = []

    for k in args.clusters:
        for run_index in range(args.num_runs):
            start = time.perf_counter()
            solution = solver.fit_predict(k=k)
            total_runtime = time.perf_counter() - start

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
                    "runtime": total_runtime,
                    "H_runtime": solution.h_runtime,
                    "fair_x": solution.fairness_violation,
                }
            )

            print(
                f"[R-FairSC, k={k}, run={run_index + 1}/{args.num_runs}] "
                f"balance={balance:.4f} "
                f"ncut={ncut:.6f} "
                f"runtime={total_runtime:.3f}s "
                f"H_time={solution.h_runtime:.3f}s "
                f"||U^T H||={solution.fairness_violation:.3e}"
            )

    summary = summarize_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Wrote summary to {output_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R-FairSC on Bail.")
    parser.add_argument(
        "--data",
        default="data/bail/bail.csv",
        help="Path to bail.csv.",
    )
    parser.add_argument(
        "--edges",
        default="data/bail/bail_edges.txt",
        help="Path to bail_edges.txt.",
    )
    parser.add_argument(
        "--sensitive-attribute",
        default="WHITE",
        help="Protected attribute column in bail.csv.",
    )
    parser.add_argument(
        "--output",
        default="results/r_fairsc_gpu_bail_fair_x.csv",
        help="Output summary CSV.",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        default=[2, 4, 6, 8, 10],
        help="Cluster counts to evaluate.",
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--alpha0", type=float, default=5e-3)
    parser.add_argument("--max-admm-iterations", type=int, default=50)
    parser.add_argument("--max-rcg-iterations", type=int, default=200)
    parser.add_argument(
        "--device",
        default=None,
        help="For example: cuda, cuda:0, or cpu. Default selects CUDA when available.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()