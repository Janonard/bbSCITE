#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter --mail-type=All --mail-user=joo@mail.upb.de -J ffscite-report
#SBATCH -p normal -q normal --time=08:00:00 --cpus-per-task=1 --mem=8G

ml fpga devel intel/oneapi/22.2.0 bittware/520n Boost/1.74.0-GCC CMake
mkdir -p build
cd build

dpcpp -std=c++20 -fintelfpga -qactypes -fsycl-link \
    -Xshardware -Xsv -Xsclock=350MHz \
    ../src/main.cpp -o ffSCITE
