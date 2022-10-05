# ffSCITE

Welcome to ffSCITE. ffSCITE (short for "fabulously fast SCITE") is an implementation of the [SCITE](https://github.com/cbg-ethz/SCITE) software package for FPGAs to analyze the output of single-cell genome sequencers and to reconstruct the mutation history of a group of cells. It has been primarily developed as part of the Bachelor's Thesis "Accelerating Single-Cell Inference of Tumor Evolution with FPGAs" by Jan-Oliver Opdenhövel, with the goal of providing an implementation of the same algorithm with a higher throughput, while maintaining the solution quality.

## About SCITE and ffSCITE

TBA: Little introduction of the problem and what makes ffSCITE "special."

## Building

ffSCITE is written in Data Parallel C++ using [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html#gs.e40rfk), targeting Intel datacenter FPGAs like the Stratix 10 family. Additionally, both the tests and the application use certain [Boost](https://www.boost.org/) libraries as well as CMake. The exact software requirements are:

* Intel oneAPI Base Toolkit, version 22.3.0 or higher
* Intel FPGA Add-on for oneAPI Base Toolkit, together with the BSP for your FPGA board
* Boost, version 1.74.0
* CMake, version 3.23.1 or higher

Some newer versions of Boost have lead to certain problems during compilation, but future release may be unaffected. On the Noctua 2 supercomputer, the required modules to target the Bittware 520N cards can be loaded with the command

``` bash
module load fpga devel intel/oneapi bittware/520n Boost/1.74.0-GCC CMake
```

### Hardware

ffSCITE with a final FPGA image is built with the command

``` bash
dpcpp -fintelfpga -qactypes -DHARDWARE \
    -Xshardware -Xsv -Xsparallel=8 -Xsprofile -Xshigh-effort -Xsseed=1 \
    -reuse-exe=./ffSCITE src/main.cpp -o ffSCITE
```

There is also the script [`scripts/synthesis.sh`](scripts/synthesis.sh) which can be submitted to `sbatch` and builds the final image.

### Emulation & Unit Tests

Building ffSCITE with the hardware emulation image, as well as building the unit tests and SCITE itself is managed with CMake. First, make sure that the SCITE module is checked out:

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

Then, the emulation version of ffSCITE is the target `ffSCITE_emu`, the unit tests are the target `unit_test`, and SCITE is the target `scite`. The unit tests (found in `./build/test/unit_test`) are simply an executable that can be run.

## Using ffSCITE

ffSCITE is CLI-compatible with SCITE and [SCITE's readme page](https://github.com/cbg-ethz/SCITE/blob/master/README.md) is applicable in most cases. The most common and required arguments are:

* `-i <input file>`: Use this input file. An input file is a CSV file with spaces as separators and entries in {0,1,2,3}. There is a row for every gene and a column for every cell, and an entry encodes whether a certain cell has a certain mutation at this gene or not. 0 encodes that a mutation is not present, 1 encodes that a (heterozygous) mutation is present, 2 encodes that a (homozygous) mutation present, and a 3 encodes that the mutation state of this cell at this gene is unknown.
* `-r <no of chain>`: Simulate the given number of monte carlo chains.
* `-l <no of chain steps>`: Execute the given number of chain steps for every chain.
* `-fd <prob>`: Assume that false positives (a mutation is reported that does not exist) occur with the given probability.
* `-ad <prob>`: Assume that false negatives (a mutation is not reported that does exist) occur with the given probability.

This means that an invocation of ffSCITE may look like this:

```
./ffSCITE -i SCITE/dataHou18.csv -r 1 -l 900000 -fd 6.04e-5 -ad 0.4309
```

ffSCITE will then load the the given input and execute the requested number of chains and steps to find the most likely mutation trees. The found trees are then written to files prefixed by the name of the input file. However, there are some differences between ffSCITE and SCITE, as well as some unsupported features:

* The chip design of ffSCITE always processes a certain group of chains at once to fully occupy the computation pipeline. The total number of chains that ffSCITE processes must therefore always be a multiple of this group size. ffSCITE automatically increases the number of chains as necessary.
* ffSCITE reads the number of cells and genes from the input file. It therefore always assumes that the dataset contains as many genes as there are rows and that there are as many cells as columns. The parameters `-n` and `-m` are therefore ignored.
* ffSCITE does not distinguish between heterozygous and homozygous mutations. Heterozygous and homozygous mutations are simply treated as mutations and are internally converted.
* Maximum aposterori (MAP) scoring is not (yet) supported. The parameter `-s` is therefore ignored.
* Attaching cells to the output trees is not (yet) supported. The parameter `-a` is therefore ignored.
* Sampling from the posterior distribution is not (yet) supported. The parameter `-p` is therefore ignored.
* Naming the mutations in the output trees is not (yet) supported. The parameter `-names` is therefore ignored.
* Setting the seed of the random number generators is not (yet) supported. The parameter `-seed` is therefore ignored.
* ffSCITE always collects the highest-scoring trees. The parameter `-no_tree_list` is therefore ignored.
* ffSCITE only works with mutation trees. Using binary leaf-labelled trees is therefore not supported and the parameter `-transpose` is ignored.

## Performance and Quality benchmarks

TBA: Describe how to run the benchmarks, how to evaluate them with the "tool."