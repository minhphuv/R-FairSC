"""Run sFairSC on the Credit dataset."""

from __future__ import annotations

import argparse

from datasets.credit import load_credit_dataset
from experiments.baseline_common import run_baseline
from solvers.sfairsc import SFairSC


def run_experiment(args: argparse.Namespace):
    dataset = load_credit_dataset(edges_path=args.edges, colors_path=args.colors)
    solver = SFairSC(dataset.adjacency, dataset.degree_inv_sqrt, dataset.fairness_matrix)
    return run_baseline(method_name="sFairSC", solver=solver, dataset=dataset, clusters=args.clusters,
                        num_runs=args.num_runs, output=args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sFairSC on Credit.")
    parser.add_argument("--edges", default="data/credit/credit_edges.csv")
    parser.add_argument("--colors", default="data/credit/credit_colors.csv")
    parser.add_argument("--output", default="results/credit_sfairsc.csv")
    parser.add_argument("--clusters", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--num-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
