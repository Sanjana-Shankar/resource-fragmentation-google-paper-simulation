"""
main.py

Entry point for the TPU resource-fragmentation simulator.

Experiments:

1. Job size vs allocation success
2. Occupancy vs allocation success
3. Failure probability vs allocation success
"""

import matplotlib.pyplot as plt

from cluster import TPUPod

from simulation import (
    experiment_job_sizes,
    experiment_occupancy,
    experiment_failures,
)


TRIALS = 10_000


def demonstrate_single_cluster():
    """
    Print one randomly generated pod so we can
    visually understand what the simulator represents.
    """

    print("\n=== Example TPU Pod ===")

    pod = TPUPod(
        failure_probability=0.02,
        occupancy_probability=0.20,
    )

    pod.print_cluster()

    print()
    print("Available cubes:", pod.available_cubes())
    print("Occupied cubes:", pod.occupied_cubes())
    print("Failed cubes:", pod.failed_cubes())
    print("Available TPUs:", pod.available_tpus())


def run_job_size_experiment():

    print("\n=== Experiment 1: Job Size ===")

    job_sizes = [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        40,
        48,
    ]

    results = experiment_job_sizes(
        job_sizes=job_sizes,
        trials=TRIALS,
        failure_probability=0.02,
        occupancy_probability=0.20,
    )

    for result in results:

        print(
            f"Job = {result['job_size']:2d} cubes | "
            f"Static = "
            f"{result['static_success_rate']:.3f} | "
            f"Reconfigurable = "
            f"{result['reconfigurable_success_rate']:.3f} | "
            f"Fragmentation blocked = "
            f"{result['fragmentation_blocked_rate']:.3f}"
        )

    x = [
        result["job_size"] * 64
        for result in results
    ]

    static = [
        result["static_success_rate"]
        for result in results
    ]

    reconfig = [
        result["reconfigurable_success_rate"]
        for result in results
    ]

    plt.figure()

    plt.plot(
        x,
        static,
        marker="o",
        label="Static allocation",
    )

    plt.plot(
        x,
        reconfig,
        marker="o",
        label="Reconfigurable allocation",
    )

    plt.xlabel("Requested Job Size (TPUs)")
    plt.ylabel("Allocation Success Rate")

    plt.title(
        "Allocation Success vs Distributed Job Size"
    )

    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("job_size_experiment.png")

    plt.show()


def run_occupancy_experiment():

    print("\n=== Experiment 2: Cluster Occupancy ===")

    occupancy_rates = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]

    requested_cubes = 16

    results = experiment_occupancy(
        occupancy_rates=occupancy_rates,
        requested_cubes=requested_cubes,
        trials=TRIALS,
        failure_probability=0.02,
    )

    for result in results:

        print(
            f"Occupancy = "
            f"{result['occupancy']:.0%} | "
            f"Static = "
            f"{result['static_success_rate']:.3f} | "
            f"Reconfigurable = "
            f"{result['reconfigurable_success_rate']:.3f}"
        )

    x = [
        result["occupancy"] * 100
        for result in results
    ]

    static = [
        result["static_success_rate"]
        for result in results
    ]

    reconfig = [
        result["reconfigurable_success_rate"]
        for result in results
    ]

    plt.figure()

    plt.plot(
        x,
        static,
        marker="o",
        label="Static allocation",
    )

    plt.plot(
        x,
        reconfig,
        marker="o",
        label="Reconfigurable allocation",
    )

    plt.xlabel("Cluster Occupancy (%)")
    plt.ylabel("Allocation Success Rate")

    plt.title(
        "Allocation Success vs Cluster Occupancy\n"
        "(16-Cube / 1024-TPU Job)"
    )

    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("occupancy_experiment.png")

    plt.show()


def run_failure_experiment():

    print("\n=== Experiment 3: Resource Failures ===")

    failure_rates = [
        0.00,
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
    ]

    requested_cubes = 16

    results = experiment_failures(
        failure_rates=failure_rates,
        requested_cubes=requested_cubes,
        trials=TRIALS,
        occupancy_probability=0.20,
    )

    for result in results:

        print(
            f"Failure probability = "
            f"{result['failure_rate']:.0%} | "
            f"Static = "
            f"{result['static_success_rate']:.3f} | "
            f"Reconfigurable = "
            f"{result['reconfigurable_success_rate']:.3f}"
        )

    x = [
        result["failure_rate"] * 100
        for result in results
    ]

    static = [
        result["static_success_rate"]
        for result in results
    ]

    reconfig = [
        result["reconfigurable_success_rate"]
        for result in results
    ]

    plt.figure()

    plt.plot(
        x,
        static,
        marker="o",
        label="Static allocation",
    )

    plt.plot(
        x,
        reconfig,
        marker="o",
        label="Reconfigurable allocation",
    )

    plt.xlabel("Cube Failure Probability (%)")
    plt.ylabel("Allocation Success Rate")

    plt.title(
        "Allocation Success vs Resource Failures\n"
        "(16-Cube / 1024-TPU Job)"
    )

    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("failure_experiment.png")

    plt.show()


def main():

    demonstrate_single_cluster()

    run_job_size_experiment()

    run_occupancy_experiment()

    run_failure_experiment()


if __name__ == "__main__":
    main()