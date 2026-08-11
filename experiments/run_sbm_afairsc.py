"""Run A-FairSC on the synthetic stochastic block model (SBM)."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from datasets.sbm import generate_sbm_dataset
from solvers.a_fairsc import AFairSC, AFairSCConfig
from utils.metrics import clustering_accuracy, compute_balance, normalized_cut


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    n_values = list(range(args.n_start, args.n_end + 1, args.n_step))

    for n in n_values:
        print(f"\n==================== n={n} ====================\n")

        for h in args.groups:
            for k in args.clusters:
                error_rates: list[float] = []
                balances: list[float] = []
                ncuts: list[float] = []
                fairness_values: list[float] = []
                runtimes: list[float] = []
                h_runtimes: list[float] = []

                print(
                    f"=== n={n}, h={h}, k={k} "
                )

                for run_index in range(args.num_runs):
                    dataset = generate_sbm_dataset(
                        n=n,
                        h=h,
                        k=k,
                        seed=args.graph_seed,
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

                    seed = args.seed + run_index
                    start = time.perf_counter()
                    solution = solver.fit_predict(k=k, seed=seed)
                    runtime = time.perf_counter() - start

                    error_rate = clustering_accuracy(
                        dataset.true_labels,
                        solution.labels,
                    )
                    balance = compute_balance(
                        cluster_labels=solution.labels,
                        group_indicator=dataset.group_indicator,
                        k=k,
                        verbose=args.balance_verbose,
                    )
                    ncut = normalized_cut(
                        dataset.adjacency,
                        solution.labels,
                    )

                    error_rates.append(error_rate)
                    balances.append(balance)
                    ncuts.append(ncut)
                    fairness_values.append(solution.fairness_violation)
                    runtimes.append(runtime)
                    h_runtimes.append(solution.h_runtime)

                    print(
                        f"  run {run_index + 1}/{args.num_runs}: "
                        f"bal={balance:.6f}, "
                        f"ncut={ncut:.6f}, "
                        f"error_rate={error_rate:.6f}, "
                        f"time={runtime:.3f}s"
                    )

                rows.append(
                    {
                        "n": int(n),
                        "h": int(h),
                        "k": int(k),
                        "alpha0": float(args.alpha0),
                        "T": int(args.max_admm_iterations),
                        "num_runs": int(args.num_runs),
                        "error_rate_mean": float(np.mean(error_rates)),
                        "error_rate_std": _sample_std(error_rates),
                        "balance_mean": float(np.mean(balances)),
                        "balance_std": _sample_std(balances),
                        "ncut_mean": float(np.mean(ncuts)),
                        "ncut_std": _sample_std(ncuts),
                        "fair_x_mean": float(np.mean(fairness_values)),
                        "fair_x_std": _sample_std(fairness_values),
                        "runtime_mean": float(np.mean(runtimes)),
                        "runtime_std": _sample_std(runtimes),
                        "H_runtime_mean": float(np.mean(h_runtimes)),
                        "H_runtime_std": _sample_std(h_runtimes),
                    }
                )

        snapshot = pd.DataFrame(rows).sort_values(["n", "h", "k"])
        snapshot_path = Path(
            f"{args.output_prefix}_up_to_n{n}.csv"
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(snapshot_path, index=False)
        print(f"Saved snapshot: {snapshot_path}")

    result = pd.DataFrame(rows).sort_values(["n", "h", "k"])
    output_path = Path(f"{args.output_prefix}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved final: {output_path}\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A-FairSC on synthetic SBM graphs."
    )
    parser.add_argument("--n-start", type=int, default=50000)
    parser.add_argument("--n-end", type=int, default=50000)
    parser.add_argument("--n-step", type=int, default=5000)
    parser.add_argument("--groups", nargs="+", type=int, default=[7, 8, 9, 10])
    parser.add_argument("--clusters", nargs="+", type=int, default=[4, 5, 6, 7])

    parser.add_argument("--alpha0", type=float, default=5e-3)
    parser.add_argument("--max-admm-iterations", type=int, default=30)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=10.0)

    parser.add_argument("--lbfgs-max-iterations", type=int, default=200)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-3)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=0.999)
    parser.add_argument("--alpha-min", type=float, default=1e-8)
    parser.add_argument("--no-warm-start", action="store_true")

    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--balance-verbose", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="results/a_fairsc_sbm",
    )
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()