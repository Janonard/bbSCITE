#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter --mail-type=All --mail-user=joo@mail.upb.de -J ffscite-synthesis
#SBATCH -p normal -q fpgasynthesis --time=2-00:00:00 --cpus-per-task=32 --mem=64G

ml fpga devel intel/oneapi bittware/520n Boost/1.74.0-GCC CMake
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j64 scite
dpcpp -fintelfpga -qactypes -Xshardware -Xsv -Xsparallel=64 ../src/main.cpp -o ffSCITE