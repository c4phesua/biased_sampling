# Advanced Algorithms Project: Biased Sampling for Narrow Passages

## 📝 Short Description
This project focuses on motion planning in continuous spaces, specifically addressing the "narrow passage" problem. Standard uniform sampling rarely hits narrow passages because the hit probability is directly proportional to the volume of the passage. 

This repository implements the Probabilistic Roadmap (PRM) algorithm and compares standard uniform sampling against heuristic biased samplers (Gaussian, Bridge test, Obstacle-based) and samplers learned from experience (CVAE, Ichter et al. 2018). The goal is to evaluate these approaches based on success rates, runtime efficiency, and the number of collision checks required to find a feasible path.

## 🛠 Tech Stack & Libraries
*   **Language:** Python 3.11
*   **Collision Checking & Geometry:** `shapely` (Core infrastructure for map and obstacle definitions)
*   **Graph Operations:** `networkx` (For pathfinding and connected components)
*   **Data Structures:** `scipy.spatial` (KDTree for efficient nearest-neighbor searches)
*   **Visualization:** `matplotlib` / `seaborn` (For plotting roadmaps and benchmark results)

---

## 📋 Task Breakdown

### 1. Core Framework & PRM Infrastructure
- [ ] Set up the 2D environment and obstacle representation (using `Shapely`).
- [ ] Implement `CollisionChecker`: Verify if a configuration node is in the free space.
- [ ] Implement `LocalPlanner(c, n)`: Check for collisions along the straight line connecting two nodes.
- [ ] Implement the neighbor search algorithm (K-Nearest Neighbors or Radius Search) to define `N_c`.
- [ ] Implement the main PRM loop (manage Node/Edge sets and check `same_connected_component`).
- [ ] Integrate a shortest-path algorithm (A* or Dijkstra) on the generated roadmap (can use `networkx`).

### 2. Sampler Implementations
- [ ] **Baseline:** Implement `UniformSampler`.
- [ ] **Heuristic 1:** Implement `GaussianSampler` (samples near obstacle boundaries).
- [ ] **Heuristic 2:** Implement `BridgeTestSampler` (designed for narrow passages).
- [ ] **Heuristic 3:** Implement `ObstacleBasedSampler` (OBPRM).
- [ ] **Learned-based:** Implement `CVAESampler` (Conditional Variational Autoencoder, per Ichter et al. 2018).

### 3. Metrics & Language-Independent Benchmarks
- [ ] Inject internal counters into the PRM class: `num_collision_checks`.
- [ ] Inject internal counters: `num_nodes_expanded`.
- [ ] Measure runtime execution for distinct phases (Sampling, Edge Connection, Pathfinding).

### 4. Benchmarking & Visualization
- [ ] Design custom 2D test maps featuring varying widths of narrow passages.
- [ ] Write an automated benchmarking script: Execute 100+ runs per sampler across different maps to calculate the success rate.
- [ ] Write a script to export benchmark results to CSV format.
- [ ] Write a visualization script (`matplotlib`/`seaborn`) to render the 2D map and the generated roadmaps.
- [ ] Generate statistical plots comparing Uniform, Heuristic, and CVAE samplers based on runtime and collision checks.