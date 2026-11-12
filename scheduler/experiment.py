"""Benchmark experiment runner for multi-agent scheduling policies."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy
from scheduler.workloads import DagName, create_workflow

DEFAULT_RESULTS_DIR = Path("scheduler/results")
DEFAULT_POLICIES = ("fifo", "cpf", "slack", "heft", "couloir")


def run_single_workflow_sweep(
    policies: Sequence[str] = DEFAULT_POLICIES,
    worker_counts: Sequence[int] = (1, 2, 3, 4),
    profile: str = "llm",
    dag: DagName = "writing",
) -> pd.DataFrame:
    """Evaluate single workflow makespan and waiting times as worker slots vary."""
    records: list[dict[str, any]] = []

    for k in worker_counts:
        for pol_name in policies:
            pol_func = get_policy(pol_name)
            wf = create_workflow(
                dag, workflow_id="wf0", profile=profile, duration_noise_std=0.0
            )
            engine = SimulationEngine(
                num_workers=k, policy_func=pol_func, policy_name=pol_name
            )
            res = engine.run([wf])
            d = res.metrics.to_dict()
            d["experiment"] = "single_workflow_sweep"
            d["dag"] = dag
            records.append(d)

    return pd.DataFrame(records)


def run_arrival_order_contention_experiment(
    profile: str = "llm",
    num_workers: int = 2,
) -> pd.DataFrame:
    """Demonstrate exact impact of dispatch ordering under K=2 (writing DAG only)."""
    records: list[dict[str, any]] = []

    # 1. Critical-Path-First (B, C, A)
    wf_cpf = create_workflow("writing", workflow_id="wf0", profile=profile)
    res_cpf = SimulationEngine(num_workers=num_workers, policy_func=get_policy("cpf"), policy_name="CPF").run([wf_cpf])
    d_cpf = res_cpf.metrics.to_dict()
    d_cpf["dispatch_order"] = "Critical-First (B, C, A)"
    records.append(d_cpf)

    # 2. Simplified HEFT
    wf_heft = create_workflow("writing", workflow_id="wf0", profile=profile)
    res_heft = SimulationEngine(num_workers=num_workers, policy_func=get_policy("heft"), policy_name="HEFT").run([wf_heft])
    d_heft = res_heft.metrics.to_dict()
    d_heft["dispatch_order"] = "HEFT Upward Rank (B, C, A)"
    records.append(d_heft)

    # 3. Least-Slack-First
    wf_slack = create_workflow("writing", workflow_id="wf0", profile=profile)
    res_slack = SimulationEngine(num_workers=num_workers, policy_func=get_policy("slack"), policy_name="Slack (LSF)").run([wf_slack])
    d_slack = res_slack.metrics.to_dict()
    d_slack["dispatch_order"] = "Least-Slack (B, C, A)"
    records.append(d_slack)

    # 4. Standard FIFO (A, B, C)
    wf_fifo = create_workflow("writing", workflow_id="wf0", profile=profile)
    res_fifo = SimulationEngine(num_workers=num_workers, policy_func=get_policy("fifo"), policy_name="FIFO (A, B, C)").run([wf_fifo])
    d_fifo = res_fifo.metrics.to_dict()
    d_fifo["dispatch_order"] = "FIFO (A, B, C)"
    records.append(d_fifo)

    # 5. Worst-case Anti-Critical FIFO (A, C, B)
    def worst_fifo_policy(ready, curr_t, wfs):
        # Dispatches smallest duration tasks first, leaving critical agent B waiting
        return sorted(ready, key=lambda t: (t.expected_duration, t.task_id))

    wf_worst = create_workflow("writing", workflow_id="wf0", profile=profile)
    res_worst = SimulationEngine(num_workers=num_workers, policy_func=worst_fifo_policy, policy_name="Anti-Critical FIFO").run([wf_worst])
    d_worst = res_worst.metrics.to_dict()
    d_worst["dispatch_order"] = "Anti-Critical (A, C, B)"
    records.append(d_worst)

    # 6. Couloir
    wf_couloir = create_workflow("writing", workflow_id="wf0", profile=profile)
    res_c = SimulationEngine(
        num_workers=num_workers, policy_func=get_policy("couloir"), policy_name="Couloir"
    ).run([wf_couloir])
    d_c = res_c.metrics.to_dict()
    d_c["dispatch_order"] = "Couloir"
    records.append(d_c)

    return pd.DataFrame(records)


def run_batch_workflow_sweep(
    policies: Sequence[str] = DEFAULT_POLICIES,
    batch_sizes: Sequence[int] = (1, 2, 5, 10, 20),
    num_workers: int = 4,
    profile: str = "llm",
    stagger_arrival_s: float = 0.0,
    dag: DagName = "writing",
) -> pd.DataFrame:
    """Evaluate multi-tenant workload makespan under fixed worker pool capacity."""
    records: list[dict[str, any]] = []

    for m in batch_sizes:
        for pol_name in policies:
            pol_func = get_policy(pol_name)
            wfs = [
                create_workflow(
                    dag,
                    workflow_id=f"wf_{i}",
                    profile=profile,
                    duration_noise_std=0.0,
                    arrival_time=i * stagger_arrival_s,
                )
                for i in range(m)
            ]
            engine = SimulationEngine(
                num_workers=num_workers,
                policy_func=pol_func,
                policy_name=pol_name,
            )
            res = engine.run(wfs)
            d = res.metrics.to_dict()
            d["experiment"] = "batch_workflow_sweep"
            d["batch_size"] = m
            d["dag"] = dag
            records.append(d)

    return pd.DataFrame(records)


def run_stochastic_robustness_sweep(
    policies: Sequence[str] = DEFAULT_POLICIES,
    noise_levels: Sequence[float] = (0.0, 0.1, 0.25, 0.5),
    num_trials: int = 30,
    num_workers: int = 2,
    profile: str = "llm",
    seed: int = 42,
    dag: DagName = "writing",
) -> pd.DataFrame:
    """Evaluate policy robustness when actual task runtimes fluctuate around mean."""
    records: list[dict[str, any]] = []

    for noise in noise_levels:
        for pol_name in policies:
            makespans = []
            wait_times = []
            cpath_delays = []

            for trial in range(num_trials):
                rng = random.Random(seed + trial * 1000 + int(noise * 100))
                wf = create_workflow(
                    dag,
                    workflow_id=f"wf_trial_{trial}",
                    profile=profile,
                    duration_noise_std=noise,
                    rng=rng,
                )
                pol_func = get_policy(pol_name)
                engine = SimulationEngine(
                    num_workers=num_workers,
                    policy_func=pol_func,
                    policy_name=pol_name,
                )
                res = engine.run([wf])
                makespans.append(res.metrics.makespan)
                wait_times.append(res.metrics.avg_waiting_time)
                cpath_delays.append(res.metrics.critical_path_delay)

            records.append(
                {
                    "experiment": "stochastic_robustness",
                    "policy": pol_name,
                    "num_workers": num_workers,
                    "noise_std": noise,
                    "num_trials": num_trials,
                    "dag": dag,
                    "mean_makespan": round(float(np.mean(makespans)), 4),
                    "std_makespan": round(float(np.std(makespans)), 4),
                    "mean_waiting_time": round(float(np.mean(wait_times)), 4),
                    "mean_cpath_delay": round(float(np.mean(cpath_delays)), 4),
                }
            )

    return pd.DataFrame(records)


def plot_benchmark_results(
    df_single: pd.DataFrame,
    df_batch: pd.DataFrame,
    df_stochastic: pd.DataFrame,
    df_order: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate publication-quality comparison charts."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    policy_colors = {
        "fifo": "#e74c3c",
        "cpf": "#2980b9",
        "slack": "#27ae60",
        "heft": "#8e44ad",
        "couloir": "#d35400",
    }
    policy_labels = {
        "fifo": "FIFO / FCFS",
        "cpf": "Critical-Path-First",
        "slack": "Least-Slack-First",
        "heft": "HEFT Upward Rank",
        "couloir": "Couloir",
    }

    # 1. Single Workflow Makespan vs Worker Slots
    ax1 = axes[0]
    for pol in df_single["policy"].unique():
        sub = df_single[df_single["policy"] == pol]
        ax1.plot(
            sub["num_workers"],
            sub["makespan"],
            marker="o",
            linewidth=2,
            label=policy_labels.get(pol, pol),
            color=policy_colors.get(pol, "black"),
        )
    ax1.set_title("Makespan vs Worker Slots (K)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Worker Slots (K)", fontsize=10)
    ax1.set_ylabel("Makespan (seconds)", fontsize=10)
    ax1.set_xticks(sorted(df_single["num_workers"].unique()))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    # 2. Dispatch Order Contention at K=2
    ax2 = axes[1]
    order_colors = ["#2980b9", "#8e44ad", "#27ae60", "#f39c12", "#c0392b"]
    bars = ax2.bar(
        df_order["policy"],
        df_order["makespan"],
        color=order_colors[: len(df_order)],
        alpha=0.85,
        edgecolor="black",
    )
    ax2.set_title("K=2 Contention: Makespan by Dispatch Order", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Makespan (seconds)", fontsize=10)
    ax2.set_ylim(0, max(df_order["makespan"]) * 1.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.tick_params(axis="x", rotation=25)

    # 3. Multi-Tenant Batch Scaling (K=4)
    ax3 = axes[2]
    for pol in df_batch["policy"].unique():
        sub = df_batch[df_batch["policy"] == pol]
        ax3.plot(
            sub["batch_size"],
            sub["makespan"],
            marker="s",
            linewidth=2,
            label=policy_labels.get(pol, pol),
            color=policy_colors.get(pol, "black"),
        )
    ax3.set_title("Batch Scaling (K=4 Slots)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Batch Size (Workflows M)", fontsize=10)
    ax3.set_ylabel("Total Makespan (seconds)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(frameon=True, fontsize=9)

    # 4. Stochastic Noise vs Makespan (K=2)
    ax4 = axes[3]
    for pol in df_stochastic["policy"].unique():
        sub = df_stochastic[df_stochastic["policy"] == pol]
        ax4.errorbar(
            sub["noise_std"],
            sub["mean_makespan"],
            yerr=sub["std_makespan"],
            marker="^",
            linewidth=2,
            capsize=4,
            label=policy_labels.get(pol, pol),
            color=policy_colors.get(pol, "black"),
        )
    ax4.set_title("Noise Robustness (K=2 Slots)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Duration Noise (Std Dev)", fontsize=10)
    ax4.set_ylabel("Mean Makespan (seconds)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    out_file = out_dir / "scheduling_comparison_plots.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved plot to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduling policy benchmark experiments.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_RESULTS_DIR), help="Output directory")
    parser.add_argument("--profile", type=str, default="llm", choices=["llm", "simulated"])
    parser.add_argument(
        "--dag",
        type=str,
        default="writing",
        choices=["writing", "chain", "fanout"],
        help="Workflow DAG shape (arrival-order experiment always uses writing)",
    )
    args = parser.parse_args()
    dag: DagName = args.dag  # type: ignore[assignment]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Experiment 1: Single Workflow Worker Limit Sweep (dag={dag})...")
    df_single = run_single_workflow_sweep(profile=args.profile, dag=dag)
    df_single.to_csv(out_dir / "single_workflow_sweep.csv", index=False)

    print("Running Experiment 2: Dispatch Order Contention at K=2 (writing DAG)...")
    df_order = run_arrival_order_contention_experiment(profile=args.profile, num_workers=2)
    df_order.to_csv(out_dir / "arrival_order_contention.csv", index=False)

    print(f"Running Experiment 3: Multi-Tenant Batch Concurrency Sweep (dag={dag})...")
    df_batch = run_batch_workflow_sweep(profile=args.profile, num_workers=4, dag=dag)
    df_batch.to_csv(out_dir / "batch_workflow_sweep.csv", index=False)

    print(f"Running Experiment 4: Stochastic Robustness Sweep (dag={dag})...")
    df_stochastic = run_stochastic_robustness_sweep(profile=args.profile, num_workers=2, dag=dag)
    df_stochastic.to_csv(out_dir / "stochastic_robustness.csv", index=False)

    print("Generating comparison charts...")
    plot_benchmark_results(df_single, df_batch, df_stochastic, df_order, out_dir)

    print(f"All benchmarks complete! Artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
