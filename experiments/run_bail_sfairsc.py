"""Run sFairSC on the Bail dataset."""

from __future__ import annotations

import argparse

from datasets.bail import load_bail_dataset
from experiments.baseline_common import run_baseline
from solvers.sfairsc import SFairSC


def run_experiment(args: argparse.Namespace):
    dataset = load_bail_dataset(data_path=args.data, edges_path=args.edges,
                                sensitive_attribute=args.sensitive_attribute)
    solver = SFairSC(dataset.adjacency, dataset.degree_inv_sqrt, dataset.fairness_matrix)
    return run_baseline(method_name="sFairSC", solver=solver, dataset=dataset, clusters=args.clusters,
                        num_runs=args.num_runs, output=args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sFairSC on Bail.")
    parser.add_argument("--data", default="data/bail/bail.csv")
    parser.add_argument("--edges", default="data/bail/bail_edges.txt")
    parser.add_argument("--sensitive-attribute", default="WHITE")
    parser.add_argument("--output", default="results/bail_sfairsc.csv")
    parser.add_argument("--clusters", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--num-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
