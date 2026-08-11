"""Run standard spectral clustering on the Diabetes dataset."""

from __future__ import annotations

import argparse

from datasets.diabetes import load_diabetes_dataset
from experiments.baseline_common import run_baseline
from solvers.standard_sc import StandardSC


def run_experiment(args: argparse.Namespace):
    dataset = load_diabetes_dataset(edges_path=args.edges, colors_path=args.colors,
                                    color_column=args.color_column)
    solver = StandardSC(dataset.adjacency)
    return run_baseline(method_name="SC", solver=solver, dataset=dataset, clusters=args.clusters,
                        num_runs=args.num_runs, output=args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SC on Diabetes.")
    parser.add_argument("--edges", default="data/diabetes/edges.csv")
    parser.add_argument("--colors", default="data/diabetes/gender.csv")
    parser.add_argument("--color-column", type=int, default=1)
    parser.add_argument("--output", default="results/diabetes_sc.csv")
    parser.add_argument("--clusters", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--num-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
