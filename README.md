# Extracting Max-Plus Weighted Finite Automata from RNNs

This repository contains the implementation of an active learning framework designed to extract Max-Plus (Tropical) Weighted Finite Automata (WFAs) directly from trained Recurrent Neural Networks (RNNs). 

## 📂 Repository Structure & Core Classes

To run any of the experiments, ensure that the following core classes are located in the same working directory as the execution notebooks:

* **`TropicalWFA.py`**: The core data structure representing the Max-Plus Automaton. It includes the methods for assigning weights to words, some reparameterizations methods (such as weight pushing and bias normalization), Viterbi pruning, and analytical graph-theory tools (e.g., Karp's Algorithm for Maximum Mean-Weight Cycles).
* **`QuantitativeObservationTable.py`**: The implementation of the L* active learning data structure tailored for the tropical semiring, including dynamic tolerance constraints and the extraction algorithms (Direct extraction and Spectral factorization).
* **`TrainedRNNOracle.py`**: The wrapper that interfaces the PyTorch RNN with the L* learner, acting as an oracle to resolve Membership and Equivalence queries.

---

## 🧪 Synthetic Case Studies (Discrete Event Systems)

The repository includes three controlled case studies built on synthetic datasets. These notebooks isolate and validate the fundamental capabilities of the extraction framework, including its robustness against stochastic noise:

1. **`StudyCase1.ipynb` (The Task Server):** Validates the framework's capability to recover 1st-order Markovian logic. It demonstrates how equivalence-preserving reparameterizations transform raw algebraic outputs into human-readable rules, how the WFA can be used to audit training shortcuts, and how the architecture handles stochastic Gaussian noise using the continuous RNN as a filter.
2. **`StudyCase2.ipynb` (The Multi-Server Problem):** Focuses on parallel synchronization and the bottleneck (maximum-cost) operator. It highlights the necessity of non-deterministic WFAs (NWFAs) to model overlapping execution paths, the use of Viterbi pruning to achieve generalization beyond the RNN's training horizon, and tests non-deterministic extraction in noisy environments.
3. **`StudyCase3.ipynb` (The Database API):** Explores 2nd-order historical dependencies. By applying Karp's Algorithm on the extracted deterministic model (DWFA), we successfully detect a system vulnerability (Maximum Asymptotic Average Cost loop) which is mathematically impossible to compute using the continuous RNN directly. It also validates the formal verification against continuous variance.

*To run these, simply open the Jupyter Notebooks and execute the cells sequentially.*

---

## 🏥 Real-World Application: OhioT1DM Dataset

This section applies the framework to model the metabolic dynamics (blood glucose variations) of a Type 1 Diabetes patient using real-world data from the OhioT1DM dataset.

While the RNN predicts glucose variation, resolving inverse clinical queries (e.g., finding the optimal combination of daily events) requires evaluating an exponential number of sequences (e.g., 4^24 permutations for a 16-hour horizon). By abstracting the RNN into a finite automaton and translating it into a Directed Acyclic Graph (DAG), we bypass this brute-force barrier, solving the queries in linear time O(|S| + |E|).

### ⚙️ How to reproduce the experiment

The complete execution environment for this application is contained within a single notebook. To run it:

1. Locate the **`WFA&RNNs`** compressed folder in the repository and extract its contents into your working directory. This folder contains:
   * The continuous PyTorch oracle (`rnn_OhioT1DM.pth`, `vocabulary_OhioT1DM.json`).
   * A set of models with the prefix `WFA_*.pkl` used to evaluate different equivalence query resolution strategies.
   * The high-precision discrete models used to answer the final clinical queries (`NWFA_651states.pkl`, `DWFA_323states.pkl`).
2. Open and run **`RealWorldApplication_OhioT1DM.ipynb`**. 
3. **Equivalence Query Strategy Comparison:** The second section of the notebook loads the specific evaluation models (`WFA_Abstraction_Big.pkl`, `WFA_Abstraction_Small.pkl`, `WFA_Exhaustive_Big.pkl`, and `WFA_Exhaustive_Small.pkl`) to evaluate the empirical predictive accuracy (MAE) of models extracted under different equivalence query methods (Exhaustive vs. Abstraction-based). It computes errors against a naive sequence-length baseline and plots the performance degradation curves over extended horizons.
4. **Solving Clinical Queries:** The middle section loads the pre-extracted WFAs to directly compute and answer three categories of clinical queries:
   * *Unconstrained Extrema* (Longest and Shortest-path DAG algorithms).
   * *Constrained Optimization* (Longest and Shortest-path DAG algorithms with additional restrictions in        the intersected DFAs).
   * *Targeted Intervals* (Integer-bounded Dynamic Programming).
5. **Active Learning Extraction:** At the bottom of the notebook, you will find the L* active learning code blocks for both non-deterministic and deterministic extractions. With the appropriate combination of configuration parameters (truncation horizons, dynamic tolerances, and quantization steps), these blocks should allow to extract the exact WFAs provided in the zip file from scratch.

## 🛠️ Requirements
- Python 3.8+
- PyTorch
- NumPy
- SciPy (for Sparse Matrices operations in graph intersections)
- Graphviz (for WFA topology visualization)
- Pandas & Scikit-learn
