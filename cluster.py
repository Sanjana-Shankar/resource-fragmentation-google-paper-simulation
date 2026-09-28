"""
cluster.py 

Simplified representaiton of a TPUv4-style accelerator pod

Real TPUv4 pod contains 4,096 TPU chips arranged into 64 cubes of 64 TPU chips each 

The simulator models each cube as a single scheduling unit 

A cube can be: 
    AVAILABLE - healthy and unused 
    OCCUPIED - currently being used by another workload 
    FAILED - unavailable because of a simulated failure
"""

import random 
from dataclasses import dataclass 
from enum import Enum 

class CubeState(Enum):
    AVAILABLE = "A"
    OCCUPIED = "O"
    FAILED = "F"

@dataclass 
class Cube:
    cube_id: int 
    state: CubeState = CubeState.AVAILABLE 

class TPUPod: 
    """
    Simplified TPU pod. 

    We model 64 cubes, with each cube representing 64 TPUs.
    """

    def __init__(
        self,
        num_cubes=64,
        tpus_per_cube=64,
        failure_probability=0.02,
        occupancy_probability=0.20,
    ): 
        self.num_cubes = num_cubes
        self.tpus_per_cube = tpus_per_cube
        self.failure_probability = failure_probability
        self.occupancy_probability = occupancy_probability

        self.cubes = []

        self._generate_cluster()
    
    def _generate_cluster(self):
        """
        Randomly assign every cube one of three states: 

        AVAILABLE 
        OCCUPIED 
        FAILED 
        """

        self.cubes = []

        for cube_id in range(self.num_cubes):

            random_number = random.random()

            if random_number == self.failure_probability:
                state = CubeState.FAILED 

            elif random_number < (
                self.failure_probability 
                + self.occupancy_probability
            ):
                state = CubeState.OCCUPIED
            
            else: 
                state = CubeState.AVAILABLE
            
            self.cubes.append(
                Cube(
                    cube_id=cube_id,
                    state=state,
                )
            )
    
    def available_cubes(self):
        """
        Return the number of currently available cubes
        """

        return sum(
            cube.state == CubeState.AVAILABLE
            for cube in self.cubes 
        )

    def occupied_cubes(self):
        """
        Return the number of occupied cubes
        """
        return sum(
            cube.state == CubeState.OCCUPIED
            for cube in self.cubes 
        )

    def failed_cubes(self):
        """
        Return the number of failed cubes
        """
        return sum(
            cube.state == CubeState.FAILED
            for cube in self.cubes 
        )
    
    def available_tpus(self):
        """
        Return the total number of available TPUs
        """
        return self.available_cubes() * self.tpus_per_cube
    
    def print_cluster(self):
        """
        Print a compact representation of the pod. 

        Example: 
        
        A A O A F A A O ....
        """

        states = [
            cube.state.value 
            for cube in self.cubes
        ]

        print(" ".join(states))

    """
    TPUPod 
        |
        |- Cube 0 -> AVAILABLE 
        |- Cube 1 -> OCCUPIED 
        | Cube 2 -> AVAILABLE 
        | Cube 3 -> FAILED 
        |_ ...
    """



    