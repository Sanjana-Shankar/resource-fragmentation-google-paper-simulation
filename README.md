# TPU Resource Fragmentation and Reconfigurable Allocation Simulator

## Overview

This project is a simplified Python simulator inspired by the resource allocation and resiliency problems discussed in:

**"Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer"**  
Yazhou Zu et al., Google, NSDI 2024.

The simulator investigates how **resource fragmentation affects the ability to schedule large distributed machine-learning jobs**.

Large distributed ML workloads may require hundreds or thousands of accelerators to be available simultaneously. Even when a cloud infrastructure has enough free accelerators in total, those resources may be unavailable to a large job because:

- some resources have failed,
- some resources are occupied by other workloads, or
- the remaining resources are fragmented in a way that does not satisfy the job's allocation constraints.

The simulator compares two simplified infrastructure models:

1. **Static contiguous allocation**
2. **TPUv4-inspired reconfigurable allocation**

The purpose of the simulator is not to reproduce Google's TPU hardware or production infrastructure exactly. Instead, it demonstrates the underlying distributed-systems principle:

> **Available capacity is not necessarily schedulable capacity.**

---

## Problem Being Simulated

The simulator focuses on:

**Resource Fragmentation and Large-Scale Gang Scheduling: Static vs. Reconfigurable TPU Allocation**

Large distributed ML workloads commonly require many accelerator resources to be allocated simultaneously.

In a statically interconnected system, sufficient total free resources may exist while a large job still cannot obtain a suitable allocation because the resources are fragmented by failures or other workloads.

For example:

```text
A = Available
O = Occupied
F = Failed

A A O A A F A A O A A A
```

A large job may have enough available resources in total but may not have a sufficiently large contiguous group under the simplified static allocation model.

A reconfigurable system can make better use of fragmented resources by combining available resources from different portions of the infrastructure.

---

## Simplified TPU Pod Model

The simulator models a TPUv4-sized resource pool consisting of:

```text
64 cubes
×
64 TPUs per cube
=
4096 TPUs
```

Each cube is treated as one scheduling unit.

Every simulated cube can have one of three states:

| State | Symbol | Meaning |
|---|---|---|
| Available | `A` | Healthy and currently unused |
| Occupied | `O` | Healthy but being used by another workload |
| Failed | `F` | Unavailable because of a simulated failure |

For example, one randomly generated cluster might look like:

```text
A A O A A A F A O A A A A O A ...
```

The simulator does not create real TPU devices, servers, virtual machines, or distributed processes. The resources are represented as Python objects so that the resource-allocation behavior can be studied without requiring cloud infrastructure.

---

## Allocation Strategies

### 1. Static Allocation

The static scheduler requires the requested number of available cubes to appear consecutively.

For example, if a job requires four cubes:

```text
A A O A A A A O
      └───────┘

4 consecutive available cubes
→ Allocation succeeds
```

However:

```text
A A O A A O A A
```

contains enough available cubes overall but does not contain four consecutive available cubes.

Therefore:

```text
Static allocation → FAIL
```

This is a simplified abstraction of the resource/topology constraints that can create fragmentation in statically interconnected accelerator infrastructure.

---

### 2. Reconfigurable Allocation

The TPUv4-inspired reconfigurable scheduler does not require the available cubes to be consecutive.

For example:

```text
A A O A A O A A
```

If the job requires four cubes, there are more than four available cubes in total.

Therefore:

```text
Reconfigurable allocation → SUCCESS
```

The model represents the high-level benefit of TPUv4's reconfigurable architecture, where available TPU cubes can be dynamically interconnected.

The simulator does **not** reproduce the actual TPUv4 Optical Circuit Switch (OCS) or Inter-Chip Interconnect (ICI) implementation.

---

## Monte Carlo Simulation

The simulator uses a **Monte Carlo simulation**.

For every trial:

1. A new 64-cube TPU pod is generated.
2. Each cube is randomly classified as available, occupied, or failed.
3. A distributed job requests a specified number of cubes.
4. The static scheduler attempts the allocation.
5. The reconfigurable scheduler attempts the same allocation.
6. The result is recorded.
7. The process is repeated thousands of times.

The default experiments use:

```text
10,000 trials
```

Repeating the experiment allows the simulator to estimate an **allocation success rate** rather than drawing conclusions from a single randomly generated cluster.

For example:

```text
8,500 successful allocations
-------------------------------- = 0.85
10,000 simulation trials
```

corresponds to an estimated allocation success rate of:

```text
85%
```

---

## Metrics

### Allocation Success Rate

The primary metric is:

```text
Allocation Success Rate =
Successful Allocations / Total Trials
```

The simulator separately calculates this metric for:

- static allocation
- reconfigurable allocation

---

### Fragmentation-Blocked Rate

The simulator also measures cases where:

```text
Reconfigurable allocation succeeds
AND
Static allocation fails
```

This means that enough resources existed in aggregate, but the static scheduler could not use them because of fragmentation.

The simulator reports this as:

```text
Fragmentation blocked
```

For example:

```text
Fragmentation blocked = 0.795
```

means that in approximately 79.5% of simulation trials, enough total resources were available for the job but the simplified static allocation constraint prevented the job from being scheduled.

---

## Experiments

The simulator performs three primary experiments.

### Experiment 1: Job Size

This experiment varies the size of the distributed job while keeping cluster occupancy and failure probability constant.

Example job sizes include:

```text
1 cube   =   64 TPUs
2 cubes  =  128 TPUs
4 cubes  =  256 TPUs
8 cubes  =  512 TPUs
16 cubes = 1024 TPUs
24 cubes = 1536 TPUs
32 cubes = 2048 TPUs
40 cubes = 2560 TPUs
48 cubes = 3072 TPUs
```

The experiment investigates:

> How does increasing distributed job size affect the probability that the required resources can be successfully allocated?

Output graph:

```text
job_size_experiment.png
```

---

### Experiment 2: Cluster Occupancy

This experiment keeps the requested job size constant while increasing the percentage of resources occupied by other workloads.

Example occupancy rates:

```text
0%
10%
20%
30%
40%
50%
60%
```

The experiment investigates:

> How does contention from other workloads increase resource fragmentation and affect the ability to schedule a large distributed job?

Output graph:

```text
occupancy_experiment.png
```

---

### Experiment 3: Resource Failures

This experiment keeps job size and cluster occupancy constant while varying the probability that resources are unavailable because of simulated failures.

Example failure probabilities:

```text
0%
1%
2%
5%
10%
15%
```

The experiment investigates:

> How does increasing resource unavailability affect static and reconfigurable resource allocation?

Output graph:

```text
failure_experiment.png
```

---

## Project Structure

The project is organized into the following files:

```text
tpu_fragmentation_simulator/
│
├── cluster.py
├── schedulers.py
├── simulation.py
├── main.py
├── requirements.txt
└── README.md
```

### `cluster.py`

Defines the simulated TPU infrastructure.

Responsibilities include:

- defining cube states,
- representing individual TPU cubes,
- creating the 64-cube TPU pod,
- randomly assigning resource states,
- counting available, occupied, and failed resources.

---

### `schedulers.py`

Implements the resource-allocation policies.

Contains:

- static contiguous allocation,
- reconfigurable allocation,
- largest available block calculation,
- fragmentation calculation.

---

### `simulation.py`

Contains the Monte Carlo simulation logic.

Responsibilities include:

- generating thousands of random infrastructure states,
- applying both allocation strategies,
- calculating allocation success rates,
- calculating fragmentation statistics,
- running experiments for different job sizes,
- running experiments for different occupancy rates,
- running experiments for different failure probabilities.

---

### `main.py`

Main entry point for the simulator.

It:

1. generates an example TPU pod,
2. runs the job-size experiment,
3. runs the occupancy experiment,
4. runs the resource-failure experiment,
5. prints experimental results to the terminal, and
6. generates graphs using Matplotlib.

---

### `requirements.txt`

Contains the external Python dependencies required to run the simulator.

The simulator currently requires:

```text
matplotlib
```

Other modules used by the simulator, such as `random`, `dataclasses`, and `enum`, are included with modern Python installations and do not need to be installed separately.

---

# Running the Simulator on Ubuntu

## 1. Install Python

Check whether Python 3 is already installed:

```bash
python3 --version
```

If Python is not installed, install it using:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

Verify the installation:

```bash
python3 --version
```

---

## 2. Navigate to the Project Directory

For example:

```bash
cd tpu_fragmentation_simulator
```

Verify that the project contains:

```bash
ls
```

You should see files similar to:

```text
README.md
cluster.py
main.py
requirements.txt
schedulers.py
simulation.py
```

---

## 3. Create a Python Virtual Environment

Creating a virtual environment is recommended so that the simulator's Python dependencies remain isolated from the system Python installation.

Create the environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

After activation, the terminal should normally display something similar to:

```text
(venv) user@ubuntu:~/tpu_fragmentation_simulator$
```

---

## 4. Install Required Packages

With the virtual environment activated, run:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

This installs Matplotlib and any other dependencies listed in `requirements.txt`.

---

## 5. Run the Simulator

Execute:

```bash
python3 main.py
```

The simulator will first display an example randomly generated TPU pod:

```text
=== Example TPU Pod ===

A A O A A A A F A O A A ...

Available cubes: 53
Occupied cubes: 10
Failed cubes: 1
Available TPUs: 3392
```

It will then execute the three experiments.

Example output:

```text
=== Experiment 1: Job Size ===

Job =  1 cubes | Static = 1.000 | Reconfigurable = 1.000
Job =  8 cubes | Static = 0.922 | Reconfigurable = 1.000
Job = 16 cubes | Static = 0.205 | Reconfigurable = 1.000
...
```

Results may vary slightly between executions because the simulator uses randomized Monte Carlo trials.

---

## 6. Generated Graphs

After running the simulator, three graphs are generated:

```text
job_size_experiment.png
occupancy_experiment.png
failure_experiment.png
```

These visualize:

```text
Job Size
    vs.
Allocation Success Rate
```

```text
Cluster Occupancy
    vs.
Allocation Success Rate
```

and:

```text
Resource Failure Probability
    vs.
Allocation Success Rate
```

Each graph compares:

```text
Static Allocation
        vs.
Reconfigurable Allocation
```

---

## Running on an Ubuntu Server Without a GUI

If the simulator is executed on a remote or headless Ubuntu server, Matplotlib may not be able to open interactive graph windows.

The graphs are already saved as PNG files by the simulator, so they can still be generated without displaying them interactively.

If necessary, set the Matplotlib backend before running:

```bash
export MPLBACKEND=Agg
python3 main.py
```

The generated PNG files can then be viewed or copied from the project directory.

---

## Re-running the Experiments

Because the simulator uses randomized resource states, results may vary slightly each time:

```bash
python3 main.py
```

The default number of trials is defined in `main.py`:

```python
TRIALS = 10_000
```

Increasing the number of trials generally reduces random variation but increases simulation runtime.

---

## Changing Simulation Parameters

The experiments can be modified in `main.py`.

Examples include changing:

```python
failure_probability=0.02
occupancy_probability=0.20
```

or changing the tested job sizes:

```python
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
```

Since each cube represents 64 TPUs:

```text
Requested TPUs = Requested Cubes × 64
```

---

# Important Simplifications and Limitations

This simulator is intended to demonstrate a **cloud/distributed-systems concept**, not reproduce Google's TPUv4 implementation.

Important simplifications include:

1. Static infrastructure is modeled using one-dimensional contiguous allocation.
2. The actual TPUv3/TPUv4 physical network topology is not simulated.
3. Optical Circuit Switches are not simulated directly.
4. ICI networking and fault-tolerant routing are not modeled.
5. Communication latency and bandwidth are not modeled.
6. Training workloads are not actually executed.
7. Occupancy is generated probabilistically rather than from real workload traces.
8. Failures are modeled as independent random events.
9. Correlated hardware failures are not modeled.
10. Google's Borg scheduler and Pod Manager are not reproduced.
11. The reconfigurable scheduler assumes that any sufficient collection of available cubes can be combined.

Therefore, the numerical results should **not** be interpreted as predictions of actual TPUv4 availability.

The simulator is intended to reproduce the qualitative systems behavior:

```text
Increasing job size
        +
Workload contention
        +
Resource failures
        ↓
Resource fragmentation
        ↓
Static allocation becomes difficult
        ↓
Reconfiguration allows more existing
capacity to remain schedulable
```

---

# Key Takeaway

The simulator demonstrates that:

> **Provisioned accelerator capacity, available accelerator capacity, and schedulable accelerator capacity are not necessarily the same.**

Large distributed jobs can fail to obtain resources even when enough free accelerators exist in aggregate because failures, competing workloads, and topology constraints can fragment the resource pool.

Reconfigurable infrastructure can reduce this stranded capacity by allowing available resources to be combined more flexibly. However, reconfiguration cannot solve actual capacity exhaustion when the number of healthy and unoccupied resources falls below the job's requirements.

This demonstrates a broader cloud-computing principle:

> **At large scale, effective resource management and failure-aware scheduling can be as important as simply provisioning additional compute resources.**

---

## Reference

Yazhou Zu et al.  
**"Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer."**  
21st USENIX Symposium on Networked Systems Design and Implementation (NSDI '24), 2024.

# AI Use Disclosure

I used ChatGPT as a supporting tool during this project. AI assistance was used to help design and organize the Python simulator, including developing the initial simulation structure, debugging and testing the static and reconfigurable allocation logic, and interpreting the resulting experimental outputs. I reviewed and tested the generated code and used the simulation results as the basis for my analysis.

I also used AI assistance while analyzing portions of the TPUv4 paper to clarify distributed-systems and cloud-computing concepts that I was less familiar with and to help connect those concepts to the simulation. In particular, AI was used to assist with understanding resource fragmentation, gang scheduling, reconfigurable infrastructure, fault tolerance, and the relationship between TPUv4's architecture and comparable concepts available through public-cloud providers such as AWS, Azure, and Google Cloud.

AI was additionally used to help organize and refine portions of the written analysis and explanations. The final conclusions, interpretation of the simulation results, selection of relevant evidence from the paper, and evaluation of how those results relate to the assignment were reviewed and incorporated by me.
