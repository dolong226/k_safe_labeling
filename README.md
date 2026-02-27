## TOWARDS SCALABLE K-SAFE LABELING VIA INCREMENTAL SAT SOLVING
This repository contains the artifacts for the paper "Towards Scalable k-Safe Labeling via Incremental
SAT Solving".

**Authors:** Huong Vu Thanh<sup>1,2</sup>, Duc Trung Kim Nguyen<sup>1</sup>, 
Long Do Duc<sup>1</sup>, Thi Huong Dao<sup>3</sup>, 
Thuan Truong Ninh<sup>1</sup>, Khanh Van To<sup>1</sup>

**Affiliation:** 
1. VNU University of Engineering and Technology, Hanoi, Vietnam
2. VMU Vietnam Maritime University, Hai Phong, Vietnam
3. Haiphong University

## Short description
The K–Labeling Problem requires assigning integer labels to
graph vertices such that adjacent vertices differ by at least K, while
minimizing the maximum used labels. This NP–hard problem has been
applied in wireless frequency assignment and networking. Efficient meth-
ods for finding exact solutions still face many limitations. This paper
investigates SAT-based techniques and provides a comparison with op-
timization methods. We propose two SAT-based encoding methods that
exploit incremental SAT solving and symmetry breaking to minimize the
span. Experiments on 16 benchmark instances show that both methods
consistently outperform a direct SAT encoding.

## Repository structure
- `dataset`: Graph instance files used for experiments.
   Each file follows a plain text format:
  - Line 1: `n m k UB` (number of vertices, edges, k value, upper bound)
  - Following lines: edge list as `u v` pairs

- `source`: Source code files for the encoding methods and configurable features.

- `results`: Experiment results.


## How to reproduce experiments
1. Clone the repository:
  ```bash
  git clone https://github.com/dolong226/k_safe_labeling
  cd k_safe_labeling
  ```

2. Add dependencies:
- Install the pysat-solver library using the command ```pip install python-sat[pblib,aiger]```
- Install IBM ILOG CPLEX and Gurobi Optimizer following the instructions on their official websites.

## Results
The `results` folder contains experiment outputs generated during evaluation.
## Citation
If you use this work, please cite:

H. V. Thanh, D. T. K. Nguyen, L. D. Duc, T. H. Dao, T. T. Ninh, and K. T. Van, "Towards Scalable k-Safe Labeling via Incremental
SAT Solving".

