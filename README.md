# Hvala: Approximate Vertex Cover Solver

![In honor of those who supported me in my final days in Serbia.](docs/serbia.jpg)

---

# The Minimum Vertex Cover Problem

The **Minimum Vertex Cover (MVC)** problem is a classic optimization problem in computer science and graph theory. It involves finding the smallest set of vertices in a graph that **covers** all edges, meaning at least one endpoint of every edge is included in the set.

## Formal Definition

Given an undirected graph $G = (V, E)$, a **vertex cover** is a subset $V' \subseteq V$ such that for every edge $(u, v) \in E$, at least one of $u$ or $v$ belongs to $V'$. The MVC problem seeks the vertex cover with the smallest cardinality.

## Importance and Applications

- **Theoretical Significance:** MVC is a well-known NP-hard problem, central to complexity theory.
- **Practical Applications:**
  - **Network Security:** Identifying critical nodes to disrupt connections.
  - **Bioinformatics:** Analyzing gene regulatory networks.
  - **Wireless Sensor Networks:** Optimizing sensor coverage.

## Related Problems

- **Maximum Independent Set:** The complement of a vertex cover.
- **Set Cover Problem:** A generalization of MVC.

---

## Problem Statement

Input: A Boolean Adjacency Matrix $M$.

Answer: Find an **approximate vertex cover** with size at most $2 \cdot \mathrm{OPT}(G)$ unconditionally (and strictly less than $2 \cdot \mathrm{OPT}(G)$ on every finite simple graph $G$, conditional on the companion theorem of the Hallelujah heuristic; see the manuscript for the full guarantees). We use the consistent name **Approximate Vertex Cover Solver** throughout this repository --- the project's title --- and reserve the phrase *minimum vertex cover* exclusively for the optimal cover ($\mathrm{OPT}(G)$), against which Hvala's output is compared. Hvala does not solve the (NP-hard) Minimum Vertex Cover problem exactly; it approximates it.

### Example Instance: 5 x 5 matrix

|        | c1  | c2  | c3  | c4  | c5  |
| ------ | --- | --- | --- | --- | --- |
| **r1** | 0   | 0   | 1   | 0   | 1   |
| **r2** | 0   | 0   | 0   | 1   | 0   |
| **r3** | 1   | 0   | 0   | 0   | 1   |
| **r4** | 0   | 1   | 0   | 0   | 0   |
| **r5** | 1   | 0   | 1   | 0   | 0   |

The input for undirected graph is typically provided in [DIMACS](http://dimacs.rutgers.edu/Challenges) format. In this way, the previous adjacency matrix is represented in a text file using the following string representation:

```
p edge 5 4
e 1 3
e 1 5
e 2 4
e 3 5
```

This represents a 5x5 matrix in DIMACS format such that each edge $(v,w)$ appears exactly once in the input file and is not repeated as $(w,v)$. In this format, every edge appears in the form of

```
e W V
```

where the fields W and V specify the endpoints of the edge while the lower-case character `e` signifies that this is an edge descriptor line.

_Example Solution:_

Vertex Cover Found `1, 3, 4`: Nodes `1`, `3`, and `4` constitute an optimal solution.

---

# Compile and Environment

## Prerequisites

- Python ≥ 3.12

## Installation

```bash
pip install hvala
```

## Execution

1. Clone the repository:

   ```bash
   git clone https://github.com/frankvegadelgado/hvala.git
   cd hvala
   ```

2. Run the script:

   ```bash
   idemo -i ./benchmarks/testMatrix1
   ```

   utilizing the `idemo` command provided by Hvala's Library to execute the Boolean adjacency matrix `hvala\benchmarks\testMatrix1`. The file `testMatrix1` represents the example described herein. We also support `.xz`, `.lzma`, `.bz2`, and `.bzip2` compressed text files.

   **Example Output:**

   ```
   testMatrix1: Vertex Cover Found 1, 3, 4
   ```

   This indicates nodes `1, 3, 4` form a vertex cover.

---

## Vertex Cover Size

Use the `-c` flag to count the nodes in the vertex cover:

```bash
idemo -i ./benchmarks/testMatrix2 -c
```

**Output:**

```
testMatrix2: Vertex Cover Size 5
```

---

# Command Options

Display help and options:

```bash
idemo -h
```

**Output:**

```bash
usage: idemo [-h] -i INPUTFILE [-a] [-b] [-c] [-v] [-l] [--version]

Compute the Approximate Vertex Cover for undirected graph encoded in DIMACS format.

options:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputFile INPUTFILE
                        input file path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a factor of at most 2
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the vertex cover
  -v, --verbose         anable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Batch Execution

Batch execution allows you to solve multiple graphs within a directory consecutively.

To view available command-line options for the `batch_idemo` command, use the following in your terminal or command prompt:

```bash
batch_idemo -h
```

This will display the following help information:

```bash
usage: batch_idemo [-h] -i INPUTDIRECTORY [-a] [-b] [-c] [-v] [-l] [--version]

Compute the Approximate Vertex Cover for all undirected graphs encoded in DIMACS format and stored in a directory.

options:
  -h, --help            show this help message and exit
  -i INPUTDIRECTORY, --inputDirectory INPUTDIRECTORY
                        Input directory path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a factor of at most 2
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the vertex cover
  -v, --verbose         anable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Testing Application

A command-line utility named `test_idemo` is provided for evaluating the Algorithm using randomly generated, large sparse matrices. It supports the following options:

```bash
usage: test_idemo [-h] -d DIMENSION [-n NUM_TESTS] [-s SPARSITY] [-a] [-b] [-c] [-w] [-v] [-l] [--version]

The Hvala Testing Application using randomly generated, large sparse matrices.

options:
  -h, --help            show this help message and exit
  -d DIMENSION, --dimension DIMENSION
                        an integer specifying the dimensions of the square matrices
  -n NUM_TESTS, --num_tests NUM_TESTS
                        an integer specifying the number of tests to run
  -s SPARSITY, --sparsity SPARSITY
                        sparsity of the matrices (0.0 for dense, close to 1.0 for very sparse)
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a factor of at most 2
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the vertex cover
  -w, --write           write the generated random matrix to a file in the current directory
  -v, --verbose         anable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Code

- Python implementation by **Frank Vega**.

---

# Complexity

```diff
+ This algorithm is an Approximate Vertex Cover Solver for the (NP-hard) Minimum Vertex Cover problem: it finds near-optimal covers in linear time, with a rigorous worst-case approximation ratio of at most 2. Experimental evidence on a 252-instance benchmark suite (Sections 6.1–6.3 of the companion manuscript) shows that the observed ratio stays below √2 ≈ 1.414 on every instance, which motivates an open question --- not a proof --- about whether Hvala can be analysed for uniform sub-√2 performance on broad restricted graph classes.
```

---

# Reproducibility Bundle

This repository ships as a full reproducibility bundle for the $252$-instance empirical study reported in the companion manuscript (Sections 6.1, 6.2, and 6.3). Everything below is sufficient to regenerate the per-family tables, the cumulative solve-time totals, and the per-instance run-times reported there.

## Repository Layout

```
hvala/
├── hvala/                       # Python package (algorithm, parsers, CLI entry points)
│   ├── algorithm.py             # Reference implementation of Hvala (Algorithm 1)
│   ├── app.py                   # `idemo` single-instance CLI
│   ├── batch.py                 # `batch_idemo` directory-scan CLI
│   ├── parser.py                # DIMACS reader (handles .mis, .clq-compliment.txt, .clq, .xz/.bz2)
│   └── utils.py                 # Helpers
├── experiment/                  # Benchmark study artefacts
│   ├── npbench/
│   │   ├── benchmarks_npbench/             # FRB sub-experiment
│   │   │   └── benchmarks_npbench.txt      # Raw log of the FRB run (41 instances; graph files excluded from git -- see Excluded Graph Files)
│   │   └── clique_complement_npbench/      # DIMACS clique-complement sub-experiment
│   │       └── clique_complement_npbench.txt  # Raw log of the DIMACS run (68 instances; graph files excluded from git)
│   ├── network/                            # Real-world sub-experiment
│   │   └── network.txt                     # Raw log of the real-world run (130 instances; graph files excluded from git)
│   └── hardest/                            # Adversarial sub-experiment (small enough to ship in full)
│       ├── crown-{25,50,100}.clq           # Crown graphs K_{k,k} minus a perfect matching
│       ├── odd-cycle-{101,501,1001}.clq    # Odd cycles
│       ├── petersen.clq                    # Petersen graph (girth 5)
│       ├── heawood.clq                     # Heawood graph (bipartite, girth 6)
│       ├── mobius-kantor.clq               # Möbius-Kantor graph (girth 6)
│       ├── pappus.clq                      # Pappus graph (bipartite, girth 6)
│       ├── desargues.clq                   # Desargues / GP(10,3) (bipartite, girth 6)
│       ├── bip-3reg-{100,500}.clq          # Random (3,3)-biregular bipartite expanders
│       └── hardest.txt                     # Raw log of the adversarial run
├── benchmarks/                  # Small example matrices used by the unit tests
├── docs/                        # Static docs (project logo etc.)
├── setup.py                     # Package metadata; pins library versions
├── pyproject.toml               # Tooling configuration
└── README.md                    # This file
```

## Graph Sources

| Family | Source | URL | License / Notes |
|---|---|---|---|
| FRB (41 instances) | NPBench § "Vertex Cover instances" (originally Ke Xu's BHOSLIB) | https://roars.dev/npbench/ | Public benchmark; planted optima distributed with the data files. |
| DIMACS clique-complement (68 instances) | NPBench § "Clique complement graphs" (originally DIMACS Second Implementation Challenge) | https://roars.dev/npbench/ | Public benchmark; OPT computed as $n-\omega(G)$ using maximum-clique values from https://www.researchgate.net/publication/336795252_Adapting_DIMACS_maximum_clique_benchmark_instances_for_the_maximum_weight_clique_problem (Mascia). $^{\dagger}$ For `C500.9` and `C1000.9`, $\omega$ is only known to be a best-known lower bound. |
| Network Data Repository (130 instances) | Cai et al., undirected simple graph collection | https://networkrepository.com/ | Public network corpus; reference cover sizes (where available) from the Milagro experiment of the companion repo. |
| Adversarial suite (13 instances, `experiment/hardest/`) | Hand-crafted by this repository | (in-repo, see `experiment/hardest/`) | All OPT values are **certified**: König matching for the bipartite cases (crown family, Heawood, Pappus, Desargues, `bip-3reg-*`), the closed-form $(n+1)/2$ for the odd cycles, and brute-force enumeration for the small non-bipartite named graphs (Petersen, Möbius-Kantor). |

## Hardware and Software

| Item | Value |
|---|---|
| CPU | Intel Core i7-1165G7 @ 2.80 GHz |
| RAM | 32 GB |
| OS | Windows 11 (single-threaded execution) |
| Python | 3.12 |
| NetworkX | 3.4.2 |
| NumPy | ≥ 2.2.1 (pinned in `setup.py`) |
| SciPy | ≥ 1.15.0 (pinned in `setup.py`) |
| Hvala package | v0.1.1 (this repository) |

Exact dependency floors are recorded in `setup.py` under `INSTALL_REQUIRES`. A `pip install hvala==0.1.1` reproduces the same package version used in the study. The pinned floors `numpy>=2.2.1`, `scipy>=1.15.0`, `networkx[default]>=3.4.2`, together with `python_requires=">=3.12"`, serve as the package lockfile for the study.

## Exact Benchmark Commands

The two NPBench sub-experiments were each launched with a single `batch_idemo` invocation from the `experiment/npbench/` directory. The `-c` flag turns on per-instance cover-size counting; the `-l` flag turns on file logging into a `.txt` log file named after the input directory.

FRB sub-experiment (41 instances):

```powershell
PS D:\Work\NP\hvala\experiment\npbench> batch_idemo -i .\benchmarks_npbench\ -c -l
```

DIMACS clique-complement sub-experiment (68 instances):

```powershell
PS D:\Work\NP\hvala\experiment\npbench> batch_idemo -i .\clique_complement_npbench\ -c -l
```

Network Data Repository run (130 instances):

```powershell
PS D:\Work\NP\hvala\experiment> batch_idemo -i .\network\ -c -l
```

Adversarial-suite run (13 instances, `experiment/hardest/`):

```powershell
PS D:\Work\NP\hvala\experiment> batch_idemo -i .\hardest\ -c -l
```

All four commands write a log file inside the input directory whose name matches the directory (`benchmarks_npbench.txt`, `clique_complement_npbench.txt`, `network.txt`, `hardest.txt`).

## Raw Output Logs

Raw per-instance logs from each batch run are committed to this repository under the same directory as the input instances:

- `experiment/npbench/benchmarks_npbench/benchmarks_npbench.txt` — FRB run log (timestamps, parse time in ms, algorithm time in ms, returned cover size).
- `experiment/npbench/clique_complement_npbench/clique_complement_npbench.txt` — DIMACS clique-complement run log (same fields).
- `experiment/network/network.txt` — Network Data Repository run log.
- `experiment/hardest/hardest.txt` — Adversarial-suite run log (same fields). On every one of the 13 adversarial instances, the cover size in this log equals the certified OPT, i.e.\ Hvala returns ratio 1.000 on every adversarial instance in the suite.

Each log line follows the format:

```
YYYY-MM-DD HH:MM:SS,sss - INFO - Parsing the Input File started
YYYY-MM-DD HH:MM:SS,sss - INFO - Parsing the Input File done in: <parse_ms> milliseconds
YYYY-MM-DD HH:MM:SS,sss - INFO - Our Algorithm with an approximate solution started
YYYY-MM-DD HH:MM:SS,sss - INFO - Our Algorithm with an approximate solution done in: <algo_ms> milliseconds
YYYY-MM-DD HH:MM:SS,sss - INFO - <instance-name>: Vertex Cover Size <size>
```

Cumulative algorithm time per sub-experiment can be regenerated by summing the `<algo_ms>` field over all instances in the corresponding log file.

## Excluded Graph Files and How to Re-download Them

To keep the committed repository to a reasonable size, **the raw DIMACS-format graph files of the three large benchmark sub-experiments are not committed to git**. Specifically, the following three folders contain only their `.txt` log file in the repository and are otherwise empty until you re-download the instances yourself:

| Folder | What is committed | What is excluded |
|---|---|---|
| `experiment/npbench/benchmarks_npbench/`         | `benchmarks_npbench.txt` (FRB run log; $41$ instances)        | $41$ `.mis` DIMACS files (FRB hard instances) |
| `experiment/npbench/clique_complement_npbench/`  | `clique_complement_npbench.txt` (DIMACS run log; $68$ instances) | $68$ `.clq-compliment.txt` DIMACS files (DIMACS clique-complement graphs) |
| `experiment/network/`                            | `network.txt` (real-world run log; $130$ instances)            | $130$ DIMACS files (Network Data Repository graphs) |

The `experiment/hardest/` folder, in contrast, is shipped **in full** --- both the $13$ `.clq` DIMACS files and the `hardest.txt` log are committed --- because the entire adversarial suite is only a few KB.

Since every per-instance numerical claim in Sections 6.1 and 6.2 of the manuscript (cover size, parse time, algorithm time) is reproducibly extracted from the committed log files, the empirical results remain fully auditable from this repository alone; only the raw graph data files themselves must be re-downloaded if you wish to rerun the experiment end-to-end.

### Re-download Instructions

**NPBench FRB ($41$ files, `experiment/npbench/benchmarks_npbench/`).** Download every `.mis` file from the NPBench § "Vertex Cover instances" page and place them in this folder. Source:

* NPBench: <https://roars.dev/npbench/>
* Direct page: NPBench § "Vertex Cover instances"

The required file names appear verbatim in the log file `benchmarks_npbench.txt` (each instance is reported on a line of the form `<name>.mis: Vertex Cover Size <size>`).

**NPBench DIMACS clique-complement ($68$ files, `experiment/npbench/clique_complement_npbench/`).** Download every `.clq-compliment.txt` file from the NPBench § "Clique complement graphs" page:

* NPBench: <https://roars.dev/npbench/>
* Direct page: NPBench § "Clique complement graphs"

The required file names appear in the log file `clique_complement_npbench.txt`. Note that two instances (`C500.9` and `C1000.9`) only have a best-known lower bound on the maximum clique number; the manuscript flags these with $^{\dagger}$ in Table 3.

**Network Data Repository ($130$ files, `experiment/network/`).** Each instance is its own page at <https://networkrepository.com/>; the page URL is `https://networkrepository.com/<name>.php`. For example, the largest single instance, `ca-coauthors-dblp` ($231$ MB uncompressed), lives at <https://networkrepository.com/ca-coauthors-dblp.php>. The full list of required file names is recorded in `network.txt` (each instance is reported on a line of the form `<name>: Vertex Cover Size <size>`).

### Reproducing the Experiments After Re-downloading

Once the graph files are in place, the original `batch_idemo` invocations from the manuscript still apply verbatim:

```powershell
PS D:\Work\NP\hvala\experiment\npbench> batch_idemo -i .\benchmarks_npbench\         -c -l
PS D:\Work\NP\hvala\experiment\npbench> batch_idemo -i .\clique_complement_npbench\  -c -l
PS D:\Work\NP\hvala\experiment>         batch_idemo -i .\network\                   -c -l
PS D:\Work\NP\hvala\experiment>         batch_idemo -i .\hardest\                   -c -l
```

The fourth command runs on the in-repository adversarial suite without any download.

### `.gitignore` Policy

The repository's `.gitignore` excludes every DIMACS-format file under `experiment/npbench/` and `experiment/network/` while keeping the `.txt` log files committed. The corresponding rules are already in `.gitignore` (`experiment/npbench/benchmarks_npbench/*.mis`, `experiment/npbench/clique_complement_npbench/*.clq-compliment.txt`, `experiment/network/<file>` for every Network Data Repository file). The `experiment/hardest/*.clq` files are explicitly **not** ignored and are committed.

## Per-Candidate Ablation

The per-family ablation reported in Section 6.3 of the manuscript runs each of the four ensemble candidates ($C_1$, $C_2$, $C_3$, and $\tilde C_4 = \text{PruneRedundant}(\tilde C_1\cup\tilde C_2\cup\tilde C_3)$) independently on every instance. The ablation harness consists of two scripts:

- `ablation.py` — single-instance, four-candidate driver. Loads a DIMACS file, runs `maximal_matching_vertex_cover`, `bucket_degree_greedy`, `covering_via_reduction_max_degree_1`, prunes each independently, then computes the pruned union, and emits a JSON record per instance.
- `aggregate.py` — collects per-instance JSON records, groups by family (FRB, brock200, brock400, ..., bio, ca, scc, socfb, ...), and computes mean cover size and "% min-attaining" rate per candidate.

Invocation (from the repository root, after `pip install hvala`):

```bash
python ablation.py experiment/npbench/benchmarks_npbench         > frb.json
python ablation.py experiment/npbench/clique_complement_npbench  > dimacs.json
python ablation.py experiment/network                            > realworld.json
python aggregate.py frb.json dimacs.json realworld.json          > ablation_summary.json
```

The summary JSON is the input used to regenerate Tables 4 and 5 of the manuscript.

## Regenerating the Manuscript Tables

A regenerator script reads the raw log files and the ablation summary JSON, and emits the LaTeX rows for every table in Sections 6.1, 6.2, and 6.3:

```bash
python make_tables.py \
    --frb-log     experiment/npbench/benchmarks_npbench/benchmarks_npbench.txt \
    --dimacs-log  experiment/npbench/clique_complement_npbench/clique_complement_npbench.txt \
    --network-log experiment/network/network.txt \
    --ablation    ablation_summary.json \
    --out         tables.tex
```

The emitted `tables.tex` is a drop-in replacement for the body rows of Tables 1–5 of the manuscript.

## Reference Values for Approximation Ratios

For the NPBench instances, the reference $\mathrm{OPT}$ values are the ones distributed by NPBench for FRB and the $n-\omega(G)$ values for DIMACS clique-complements (with $\omega$ from Mascia's compilation, with the two $^{\dagger}$-flagged best-known-only cases noted above). For the Network Data Repository instances, the reference cover sizes (51 of 130 instances) are taken from the Milagro experiment compilation; of those 51, 29 are mathematically certified optima recovered from tree-like or otherwise polynomially solvable components, and 22 are best-known approximate cover sizes. For the adversarial `experiment/hardest/` suite, all 13 instances have a **certified** $\mathrm{OPT}$ computed in one of three ways: by König's theorem (maximum matching equals minimum vertex cover) on the bipartite instances (crown family, Heawood, Pappus, Desargues, both `bip-3reg-*` expanders); by the closed-form formula $\mathrm{OPT}(C_n)=(n+1)/2$ for odd cycles; and by exhaustive brute force on the small non-bipartite named graphs (Petersen, Möbius-Kantor). The manuscript reports certified approximation ratios and ratios-to-best-known separately; the same separation is recorded per-instance in the corresponding result tables.

---

# License

- MIT License.
