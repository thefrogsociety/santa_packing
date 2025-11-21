🌲 Santa 2025 Tree Packing Solver (Simulated Annealing & Multiprocessing)

This project implements a high-performance solver for the tree packing problem, aiming to place $N$ unique tree shapes within the smallest possible square boundary without overlaps. It utilizes a highly optimized Simulated Annealing (SA) core and parallelizes the workload across multiple CPU cores using Python's multiprocessing.

🚀 Key Features

Multiprocessing: Solves multiple $N$ values concurrently to maximize throughput.

Incremental Cost Calculation: The core Simulated Annealing loop operates in $O(N)$ time complexity per iteration, ensuring fast convergence even for $N=200$.

Adaptive Shrink Strategy: Uses an aggressive shrink/retry loop to find the theoretical minimum box size for each $N$.

Optimized Start: Uses a lattice initialization mode for large $N$ to quickly find a valid, dense starting configuration.

📁 Project Structure

The solution is organized into three primary modules:

File

Description

solve_all.py

Main Entry Point. Handles file I/O, multiprocessing setup, task queuing (resuming from saved state), and coordinating the optimization process for $N=1$ to $N=200$.

optimizer.py

Core Solver Logic. Contains the PackingSolver class, implementing the Simulated Annealing (SA) algorithm, incremental cost functions (overlap, boundary, compression, pull), and the final validity checks (check_valid).

geometry.py

Tree Definitions. Defines the precise coordinates of the unique tree shape and provides robust, fast utility functions for coordinate transformation (rotation/translation) and generating Shapely Polygon objects.

submission_final_run.csv

Output file containing the final calculated coordinates and rotations for all solved $N$ values.

🛠️ Setup and Dependencies

This project requires Python 3.8+ and the following libraries:

pip install numpy pandas shapely tqdm


▶️ How to Run

Execute the main script. It will automatically load any previous results from submission_final_run.csv and begin solving the remaining problems concurrently.

python solve_all.py


Multiprocessing & Stability Notes

The output verbosity is automatically suppressed within worker processes to prevent I/O contention and freezing, but progress is reported for the main shrink/retry stages.

The system dynamically reduces the number of retries for small $N$ (e.g., $N \le 3$) to prevent unnecessary, long runs on trivial cases.

⚙️ Optimization Strategy Highlights

Iterative Squeeze: The main loop repeatedly attempts to pack the previous best solution into a box reduced by a SHRINK_RATE (e.g., 0.98 or 0.99).

Simulated Annealing (SA): The PackingSolver uses SA to resolve the overlaps in the smaller box. It applies a mix of small position/rotation perturbations (99% chance) and occasional large coordinate swaps (1% chance) to escape local minima.

Cost Function Components:

Overlap Penalty (Dominant): Penalizes the shared area between intersecting tree polygons heavily.

Boundary Penalty: Penalizes any part of a tree that lies outside the defined square box boundaries.

Compression Cost: Directly minimizes the overall maximum $x$ and $y$ extent of the packing, which is the competition's final metric.

Hole Pull (Optional): Applies a gravity-like penalty to pull trees toward a specific "hole" coordinate if one is identified during runtime (e.g., for complex large-$N$ fine-tuning).
