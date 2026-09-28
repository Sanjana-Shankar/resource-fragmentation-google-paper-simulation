"""
schedulers.py

Scheduling policies used by the simulator.

We compare: 

1. Static contiguous allocation 
2. TPUv4-like reconfigurable allocation 

These are simplified models intended to demonstrate resource fragmentation.
"""

from cluster import CubeState 

def static_allocate(pod, requested_cubes):
    """
    Static allocation model.

    The requested job can run only if we can find requested_cubes AVAILABLE cubes consecutively.

    Example: 

    Request = 4 cubes 

    A A O A A A A O -> 4 available = success 

    But: 
    A A O A A O A A 
    
    contains enough total available cubes but no contiguos block of four, so allocation fails.
    """

    consecutive_available = 0

    for cube in pod.cubes: 
        if cube.state == CubeState.AVAILABLE:
            consecutive_available += 1

            if consecutive_available >= requested_cubes: 
                return True 
        else: 
            consecutive_available = 0
    
    return False 

def reconfigurable_allocate(pod, requested_cubes):
    """
    Simplified TPUv4-like allocation. 

    Available cubes do not have to be physically contiguous.

    We therefore only need to determine whether enough healthy/unoccupied cubes exist
    """
    return pod.available_cubes() >= requested_cubes 

def largest_available_block(pod):
    """
    Find the largest contiguous block of available cubes. 
    
    Used to quantify fragmentation. 
    """
    largest_block = 0
    current_block = 0 

    for cube in pod.cubes: 
        if cube.state == CubeState.AVAILABLE: 
            current_block += 1

            largest_block = max(
                largest_block,
                current_block,
            )
        else:
            current_block = 0
    return largest_block 

def fragmentation_score(pod):
    """
    Simplified fragmentation metric: 
        fragmentation = 
            1 - largest_contiguous_block / total_available 
    0 means the available capactiy is completely contiguous 

    Values closer to 1 indicate that available capacity is highly fragmented 

    Simulator defined metric: Not a metric taken directly form the Google TPUv4 paper.
    """
    total_available = pod.available_cubes()

    if total_available == 0:
        return 0.0
    
    largest_block = largest_available_block(pod)

    return 1 - (
        largest_block / total_available 
    )

