"""Run A-FairSC on the Bail dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from datasets.bail import load_bail_dataset
from solvers.a_fairsc import AFairSC, AFairSCConfig
from utils.metrics import compute_balance, normalized_cut


def _sample_std(values: np.ndarray) -> float:
    return float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_results(results: list[dict]) -> pd.DataFrame:
    """Aggregate repeated A-FairSC runs for each number of clusters."""
    rows = []

    for k in sorted({int(result["k"]) for result in results}):
        runs = [result for result in results if int(result["k"]) == k]

        balances = np.asarray([run["balance"] for run in runs], dtype=float)
        ncuts = np.asarray([run["ncut"] for run in runs], dtype=float)
        runtimes = np.asarray([run["runtime"] for run in runs], dtype=float)
        h_runtimes = np.asarray(
            [run["H_runtime"] for run in runs], dtype=float
        )
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

    print(
        f"Bail largest connected component: "
        f"{dataset.adjacency.shape[0]} nodes, "
        f"{dataset.adjacency.nnz // 2} edges"
    )

    for group_value, count in zip(
        dataset.group_values,
        dataset.group_counts,
    ):
        print(
            f"size of {args.sensitive_attribute} group "
            f"{group_value}: {count}"
        )

    config = AFairSCConfig(
        alpha0=args.alpha0,
        max_admm_iterations=args.max_admm_iterations,
        tau=args.tau,
        mu=args.mu,
        lbfgs_max_iterations=args.lbfgs_max_iterations,
        lbfgs_gtol=args.lbfgs_gtol,
        lbfgs_ftol=args.lbfgs_ftol,
        warm_start_dual=not args.no_warm_start,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
    )

    solver = AFairSC(
        adjacency=dataset.adjacency,
        degree_inv_sqrt=dataset.degree_inv_sqrt,
        fairness_matrix=dataset.fairness_matrix,
        config=config,
        verbose=args.verbose,
    )

    results: list[dict] = []

    for k in args.clusters:
        for run_index in range(args.num_runs):
            seed = args.seed + run_index

            print(
                f"\n==== k={k} "
                f"(run {run_index + 1}/{args.num_runs}) ===="
            )

            start = time.perf_counter()
            solution = solver.fit_predict(
                k=k,
                seed=seed,
            )
            total_runtime = time.perf_counter() - start

            balance = compute_balance(
                cluster_labels=solution.labels,
                group_indicator=dataset.group_indicator,
                k=k,
                verbose=False,
            )

            ncut = normalized_cut(
                dataset.adjacency,
                solution.labels,
            )

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
                f"[A-FairSC, k={k}, "
                f"run={run_index + 1}/{args.num_runs}] "
                f"balance={balance:.4f} "
                f"ncut={ncut:.6f} "
                f"runtime={total_runtime:.3f}s "
                f"H_time={solution.h_runtime:.3f}s "
                f"bins={solution.cluster_sizes} "
                f"fair_x={solution.fairness_violation:.3e}"
            )

    summary = summarize_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_path, index=False)

    print(f"\nWrote summary to {output_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A-FairSC on Bail.")

    parser.add_argument("--data", default="data/bail/bail.csv", help="Path to bail.csv.")
    parser.add_argument("--edges", default="data/bail/bail_edges.txt", help="Path to bail_edges.txt.")
    parser.add_argument("--sensitive-attribute", default="WHITE", help="Sensitive attribute column.")
    parser.add_argument("--output", default="results/baselines_bail_afairsc_time_fair_x.csv")
    parser.add_argument("--clusters", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--alpha0", type=float, default=5e-3)
    parser.add_argument("--max-admm-iterations", type=int, default=50)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--lbfgs-max-iterations", type=int, default=200)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-3)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=0.99)
    parser.add_argument("--alpha-min", type=float, default=1e-8)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()