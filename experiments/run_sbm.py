"""Run R-FairSC on the synthetic stochastic block model (SBM)."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from datasets.sbm import generate_sbm_dataset
from solvers.r_fairsc import RFairSC, RFairSCConfig
from utils.metrics import clustering_accuracy, compute_balance, normalized_cut


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    all_rows: list[dict] = []
    n_values = list(range(args.n_start, args.n_end + 1, args.n_step))

    for n in n_values:
        print(f"\n==================== n={n} ====================\n")

        for h in args.groups:
            for k in args.clusters:
                error_rates: list[float] = []
                ncuts: list[float] = []
                balances: list[float] = []
                runtimes: list[float] = []

                print(f"=== n={n}, h={h}, k={k} ===")

                for run_index in range(args.num_runs):
                    dataset = generate_sbm_dataset(
                        n=n,
                        h=h,
                        k=k,
                        seed=args.graph_seed,
                    )

                    config = RFairSCConfig(
                        alpha0=args.alpha0,
                        max_admm_iterations=args.max_admm_iterations,
                        tau=args.tau,
                        mu=args.mu,
                        max_rcg_iterations=args.max_rcg_iterations,
                        min_step_size=args.min_step_size,
                        grad_tol=args.grad_tol,
                        primal_tol=args.primal_tol,
                        fairness_tol=args.fairness_tol,
                    )

                    solver = RFairSC(
                        adjacency=dataset.adjacency,
                        degree_inv_sqrt=dataset.degree_inv_sqrt,
                        fairness_matrix=dataset.fairness_matrix,
                        config=config,
                        device=args.device,
                        verbose=args.verbose,
                    )

                    start = time.perf_counter()
                    solution = solver.fit_predict(k=k)
                    runtime = time.perf_counter() - start
                    runtimes.append(runtime)

                    message = [f"time={runtime:.3f}s"]

                    if args.compute_error_rate:
                        error_rate = clustering_accuracy(
                            dataset.true_labels,
                            solution.labels,
                        )
                        error_rates.append(error_rate)
                        message.insert(0, f"error_rate={error_rate:.6f}")

                    if args.compute_ncut:
                        ncut = normalized_cut(
                            dataset.adjacency,
                            solution.labels,
                        )
                        ncuts.append(ncut)
                        message.insert(0, f"ncut={ncut:.6f}")

                    if args.compute_balance:
                        balance = compute_balance(
                            cluster_labels=solution.labels,
                            group_indicator=dataset.group_indicator,
                            k=k,
                            verbose=args.balance_verbose,
                        )
                        balances.append(balance)
                        message.insert(0, f"bal={balance:.6f}")

                    print(
                        f"  run {run_index + 1}/{args.num_runs}: "
                        + ", ".join(message)
                    )

                row = {
                    "n": int(n),
                    "h": int(h),
                    "k": int(k),
                    "alpha0": float(args.alpha0),
                    "T": int(args.max_admm_iterations),
                    "num_runs": int(args.num_runs),
                    "time_mean": float(np.mean(runtimes)),
                    "time_std": _sample_std(runtimes),
                }

                if args.compute_error_rate:
                    row.update(
                        error_rate_mean=float(np.mean(error_rates)),
                        error_rate_std=_sample_std(error_rates),
                    )
                if args.compute_ncut:
                    row.update(
                        ncut_mean=float(np.mean(ncuts)),
                        ncut_std=_sample_std(ncuts),
                    )
                if args.compute_balance:
                    row.update(
                        balance_mean=float(np.mean(balances)),
                        balance_std=_sample_std(balances),
                    )

                all_rows.append(row)

        snapshot = pd.DataFrame(all_rows)
        snapshot_path = Path(f"{args.output_prefix}_up_to_n{n}.csv")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(snapshot_path, index=False)
        print(f"Saved snapshot: {snapshot_path}")

    result = pd.DataFrame(all_rows).sort_values(["n", "h", "k"])
    output_path = Path(f"{args.output_prefix}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved final: {output_path}\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R-FairSC on synthetic SBM graphs.")
    parser.add_argument("--n-start", type=int, default=50000)
    parser.add_argument("--n-end", type=int, default=50000)
    parser.add_argument("--n-step", type=int, default=5000)
    parser.add_argument("--groups", nargs="+", type=int, default=[7, 8, 9, 10])
    parser.add_argument("--clusters", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--alpha0", type=float, default=5e-3)
    parser.add_argument("--max-admm-iterations", type=int, default=30)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-prefix", default="results/r_fairsc_sbm")

    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--max-rcg-iterations", type=int, default=200)
    parser.add_argument("--min-step-size", type=float, default=1e-5)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--primal-tol", type=float, default=1e-4)
    parser.add_argument("--fairness-tol", type=float, default=1e-4)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--balance-verbose", action="store_true")

    parser.add_argument("--no-error-rate", dest="compute_error_rate", action="store_false")
    parser.add_argument("--no-ncut", dest="compute_ncut", action="store_false")
    parser.add_argument("--no-balance", dest="compute_balance", action="store_false")
    parser.set_defaults(compute_error_rate=True, compute_ncut=True, compute_balance=True)

    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()