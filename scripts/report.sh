#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter --mail-type=All --mail-user=joo@mail.upb.de -J ffscite-report
#SBATCH -p normal --time=08:00:00 --cpus-per-task=1 --mem=8G

ml fpga devel intel/oneapi/22.2.0 bittware/520n Boost/1.79.0 CMake
mkdir -p build
cd build

time dpcpp -std=c++20 -fintelfpga -qactypes -fsycl-link -Xshardware -Xsv -Xsclock=200MHz  \
    ../src/main.cpp -o ffSCITE
