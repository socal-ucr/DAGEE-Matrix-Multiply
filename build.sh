#!/usr/bin/env bash 

INSTALL_DIR=${HOME}/.opt/rocm
cwd=$(pwd)

git submodule update --init --recursive

rocBLAS_DIR="$(readlink -f rocBLAS)"
Tensile_DIR="$(readlink -f Tensile)"
ATMI_DIR="$(readlink -f atmi)"
DAGEE_DIR="$(readlink -f DAGEE)"

cd ${rocBLAS_DIR}
rm -rf build && mkdir build && cd build
CXX=${INSTALL_DIR}/hip/bin/hipcc /usr/local/bin/cmake -DHIP_RUNTIME=rocclr -DROCM_PATH=${INSTALL_DIR}/rocm -DTensile_LOGIC=hip_lite -DTensile_ARCHITECTURE=gfx906:xnack-  -DTensile_CODE_OBJECT_VERSION=V3 -DCMAKE_BUILD_TYPE=Release -DTensile_TEST_LOCAL_PATH=${Tensile_DIR} -DBUILD_WITH_TENSILE_HOST=ON -DTensile_LIBRARY_FORMAT=msgpack -DRUN_HEADER_TESTING=OFF -DTensile_COMPILER=hipcc -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCPACK_PACKAGING_INSTALL_PREFIX=${INSTALL_DIR} ..
make -j
make install

wget https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_13.0-5/aomp_Ubuntu1804_13.0-5_amd64.deb
dpkg -x aomp_Ubuntu1804_13.0-5_amd64.deb ${INSTALL_DIR}/aomp


cd ${ATMI_DIR}
cd src
rm -rf build && mkdir build && cd build
CXX=${INSTALL_DIR}/bin/hipcc /usr/local/bin/cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/atmi -DROCM_ROOT=${INSTALL_DIR} -DHSA_ROOT=${INSTALL_DIR} -DAMD_LLVM=${INSTALL_DIR}/aomp -DGFX_VER=gfx906 -DATMI_ROOT=/home/mchow009/.opt/rocm/atmi ..
make -j
make install


