# bbSCITE

Welcome to bbSCITE. bbSCITE (short for "bit-based SCITE") is an FPGA-accelerated implementation of the [SCITE](https://github.com/cbg-ethz/SCITE) software package to compute mutation histories of somatic cells. Given noisy mutation profiles of single cells, bbSCITE performs a stochastic search to find the Maximum Likelihood (ML) tree. Tree reconstruction can be combined with an estimation of the error rates in the mutation profiles.

The SCITE algorithm is particularly designed for reconstructing mutation histories of tumors based on mutation profiles obtained from single-cell exome sequencing experiments, but is in principle applicable to any type of (noisy) mutation profiles for which the infinite sites assumption can be made.
 
This implementation in particular has been developed as part of the Bachelor's Thesis "Accelerating Single-Cell Inference of Tumor Evolution with FPGAs" by Jan-Oliver Opdenhövel, with the goal of providing an implementation of the same algorithm with a higher throughput, while maintaining the solution quality. The LaTeX code of the thesis is found in [`docs/thesis/thesis.tex`](docs/thesis/thesis.tex) and it can be built using a full [TeX-Live](https://tug.org/texlive/) installation.

bbSCITE is open source software and available under the GPL3 license.

## Building

bbSCITE is written in Data Parallel C++ using [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html#gs.e40rfk), targeting Intel datacenter FPGAs like the Stratix 10 family. Additionally, both the tests and the application use certain [Boost](https://www.boost.org/) libraries as well as CMake. The exact software requirements are:

* Intel oneAPI Base Toolkit, version 22.3.0 or higher
* Intel FPGA Add-on for oneAPI Base Toolkit, together with the BSP for your FPGA board
* Boost, version 1.79.0
* CMake, version 3.23.1 or higher

Some newer versions of Boost have lead to certain problems during compilation, but future release may be unaffected. On the Noctua 2 supercomputer, the required modules to target the Bittware 520N cards can be loaded with the command

``` bash
module load fpga devel intel/oneapi bittware/520n Boost/1.79.0-GCC CMake
```

### Hardware

bbSCITE with a final FPGA image is built with the command

``` bash
dpcpp -fintelfpga -qactypes -DHARDWARE \
    -Xshardware -Xsv -Xsparallel=8 -Xsprofile -Xshigh-effort -Xsseed=1 \
    -reuse-exe=./bbSCITE src/main.cpp -o bbSCITE
```

There is also the script [`scripts/synthesis.sh`](scripts/synthesis.sh) which can be submitted to `sbatch` and builds the final image.

### Emulation & Unit Tests

Building bbSCITE with the hardware emulation image, as well as building the unit tests and SCITE itself is managed with CMake. First, make sure that the SCITE module is checked out:

``` bash
git submodule init
git submodule update
```

Then, create the `build` directory and generate the build files:

``` bash
mkdir build
cd build
cmake ..
```

Then, the emulation version of bbSCITE is the target `bbSCITE_emu`, the unit tests are the target `unit_test`, and SCITE is the target `scite`. The unit tests (found in `./build/test/unit_test`) are simply an executable that can be run.

## Using bbSCITE

bbSCITE is CLI-compatible with SCITE and [SCITE's readme page](https://github.com/cbg-ethz/SCITE/blob/master/README.md) is applicable in most cases. The most common and required arguments are:

* `-i <input file>`: Use this input file. An input file is a CSV file with spaces as separators and entries in {0,1,2,3}. There is a row for every gene and a column for every cell, and an entry encodes whether a certain cell has a certain mutation at this gene or not. 0 encodes that a mutation is not present, 1 encodes that a (heterozygous) mutation is present, 2 encodes that a (homozygous) mutation present, and a 3 encodes that the mutation state of this cell at this gene is unknown.
* `-r <no of chain>`: Simulate the given number of monte carlo chains.
* `-l <no of chain steps>`: Execute the given number of chain steps for every chain.
* `-fd <prob>`: Assume that false positives (a mutation is reported that does not exist) occur with the given probability.
* `-ad <prob>`: Assume that false negatives (a mutation is not reported that does exist) occur with the given probability.

This means that an invocation of bbSCITE may look like this:

```
./bbSCITE -i SCITE/dataHou18.csv -r 1 -l 900000 -fd 6.04e-5 -ad 0.4309
```

bbSCITE will then load the the given input and execute the requested number of chains and steps to find the most likely mutation trees. The found trees are then written to files prefixed by the name of the input file. However, there are some differences between bbSCITE and SCITE, as well as some unsupported features:

* The chip design of bbSCITE always processes a certain group of chains at once to fully occupy the computation pipeline. The total number of chains that bbSCITE processes must therefore always be a multiple of this group size. bbSCITE automatically increases the number of chains as necessary.
* bbSCITE reads the number of cells and genes from the input file. It therefore always assumes that the dataset contains as many genes as there are rows and that there are as many cells as columns. The parameters `-n` and `-m` are therefore ignored.
* bbSCITE does not distinguish between heterozygous and homozygous mutations. Heterozygous and homozygous mutations are simply treated as mutations and are internally converted.
* Maximum aposterori (MAP) scoring is not (yet) supported. The parameter `-s` is therefore ignored.
* Attaching cells to the output trees is not (yet) supported. The parameter `-a` is therefore ignored.
* Sampling from the posterior distribution is not (yet) supported. The parameter `-p` is therefore ignored.
* Naming the mutations in the output trees is not (yet) supported. The parameter `-names` is therefore ignored.
* Setting the seed of the random number generators is not (yet) supported. The parameter `-seed` is therefore ignored.
* bbSCITE always collects the highest-scoring trees. The parameter `-no_tree_list` is therefore ignored.
* bbSCITE only works with mutation trees. Using binary leaf-labelled trees is therefore not supported and the parameter `-transpose` is ignored.

## Performance and Quality benchmarks

The following section describes the execution of the performance and quality benchmarks referenced in the thesis. First of all, all scripts assume that a valid bbSCITE binary found under the path `build/bbSCITE` and that a valid SCITE binary is found under the path `build/scite`. The scripts are also specifically tailored to the environment of the Noctua 2 supercomputer, the targeted Bittware 520N cards, and one specific user accounts. Running the benchmarks on other systems, for other cards, or other other user accounts may require changes.

The performance benchmark is prepared and submitted as follows:
``` bash
./scripts/performance_benchmark/generate_inputs.sh
sbatch scripts/performance_benchmark/ffscite.sh
sbatch scripts/performance_benchmark/scite.sh
```
This creates the output directory `performance_benchmark.out` with different inputs and submits the execution jobs to the workload manager: One for the FPGA-based bbSCITE and one for the single-threaded SCITE. After these jobs are finished, the outputs can be analyzed with the verification tool:
``` bash
module load fpga intel/oneapi # Load oneAPI, which contains a Python distribution with all required libraries installed.
./tool.py quickperf # Print some quick information of the performance data
./tool.py perftable # Print a markdown table with performance data
./tool.py perfgraph # Visualize the performance data with Matplotlib
```
These subcommands have multiple options which can be found by executing them with the `-h` option.

The quality benchmark works similarly. Preparation and submission is done with the following commands:
``` bash
./scripts/quality_benchmark/generate_inputs.sh
sbatch scripts/quality_benchmark/ffscite.sh
sbatch scripts/quality_benchmark/scite.sh
```
Once these jobs are completed, the results can be retrieved with the following command:
``` bash
module load fpga intel/oneapi # Load oneAPI, which contains a Python distribution with all required libraries installed.
./tool.py tost # Execute the "Two One-Sided t-Tests" procedure on the sampled quality data.
```