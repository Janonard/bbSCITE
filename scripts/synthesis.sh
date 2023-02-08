#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter --mail-type=All --mail-user=joo@mail.upb.de -J ffscite-synthesis
#SBATCH -p normal -q fpgasynthesis --time=2-00:00:00 --cpus-per-task=8 --mem=200G

ml fpga devel intel/oneapi/22.3.0 bittware/520n Boost CMake
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 scite
make ffSCITE
tar -caf build.tar.gz scite ffSCITE ffSCITE.prj/reports
