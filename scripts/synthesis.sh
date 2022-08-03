#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter --mail-type=All --mail-user=joo@mail.upb.de -J ffscite-synthesis
#SBATCH -p normal -q fpgasynthesis --time=1-00:00:00 --cpus-per-task=8 --mem=200G

ml fpga devel intel/oneapi bittware/520n Boost/1.74.0-GCC CMake
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 scite
dpcpp -fintelfpga -qactypes -DHARDWARE -Xshardware -Xsv -Xsparallel=8 -Xsprofile -reuse-exe=./ffSCITE ../src/main.cpp -o ffSCITE
tar -caf build.tar.gz scite ffSCITE ffSCITE.prj/reports
