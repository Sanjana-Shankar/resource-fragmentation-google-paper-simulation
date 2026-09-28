"""
simulation.py

Monte Carlo simulation engine.

Each trial:

1. Generates a random TPU pod.
2. Applies the static scheduler.
3. Applies the reconfigurable scheduler.
4. Records whether each scheduler succeeded.
5. Measures fragmentation.

Repeating this thousands of times lets us estimate
allocation success probability.
"""

from cluster import TPUPod
from schedulers import (
    static_allocate,
    reconfigurable_allocate,
    fragmentation_score,
)


def run_simulation(
    trials,
    requested_cubes,
    failure_probability,
    occupancy_probability,
):
    """
    Run one experimental configuration.

    Returns statistics for static and
    reconfigurable scheduling.
    """

    static_successes = 0
    reconfigurable_successes = 0

    total_fragmentation = 0.0

    free_but_static_failed = 0

    for _ in range(trials):

        pod = TPUPod(
            failure_probability=failure_probability,
            occupancy_probability=occupancy_probability,
        )

        static_success = static_allocate(
            pod,
            requested_cubes,
        )

        reconfig_success = reconfigurable_allocate(
            pod,
            requested_cubes,
        )

        if static_success:
            static_successes += 1

        if reconfig_success:
            reconfigurable_successes += 1

        # Fragmentation condition:
        #
        # Enough resources exist overall,
        # but static allocation cannot use them.

        if (
            reconfig_success
            and not static_success
        ):
            free_but_static_failed += 1

        total_fragmentation += fragmentation_score(pod)

    return {
        "static_success_rate":
            static_successes / trials,

        "reconfigurable_success_rate":
            reconfigurable_successes / trials,

        "average_fragmentation":
            total_fragmentation / trials,

        "fragmentation_blocked_rate":
            free_but_static_failed / trials,
    }


def experiment_job_sizes(
    job_sizes,
    trials=10000,
    failure_probability=0.02,
    occupancy_probability=0.20,
):
    """
    Run simulations while changing job size.
    """

    results = []

    for job_size in job_sizes:

        result = run_simulation(
            trials=trials,
            requested_cubes=job_size,
            failure_probability=failure_probability,
            occupancy_probability=occupancy_probability,
        )

        result["job_size"] = job_size

        results.append(result)

    return results


def experiment_occupancy(
    occupancy_rates,
    requested_cubes,
    trials=10000,
    failure_probability=0.02,
):
    """
    Run simulations while changing cluster occupancy.
    """

    results = []

    for occupancy in occupancy_rates:

        result = run_simulation(
            trials=trials,
            requested_cubes=requested_cubes,
            failure_probability=failure_probability,
            occupancy_probability=occupancy,
        )

        result["occupancy"] = occupancy

        results.append(result)

    return results


def experiment_failures(
    failure_rates,
    requested_cubes,
    trials=10000,
    occupancy_probability=0.20,
):
    """
    Run simulations while changing failure probability.
    """

    results = []

    for failure_rate in failure_rates:

        result = run_simulation(
            trials=trials,
            requested_cubes=requested_cubes,
            failure_probability=failure_rate,
            occupancy_probability=occupancy_probability,
        )

        result["failure_rate"] = failure_rate

        results.append(result)

    return results