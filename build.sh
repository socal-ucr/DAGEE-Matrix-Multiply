#!/usr/bin/env bash 

INSTALL_DIR=${HOME}/.opt
cwd=$(pwd)

git submodule update --init --recursive

rocBLAS_DIR="$(readlink -f rocBLAS)"
Tensile_DIR="$(readlink -f Tensile)"

cd ${rocBLAS_DIR}
rm -rf build && mkdir build && cd build
CXX=${INSTALL_DIR}/rocm/hip/bin/hipcc cmake -DHIP_RUNTIME=rocclr -DROCM_PATH=${INSTALL_DIR}/rocm -DTensile_LOGIC=hip_lite -DTensile_ARCHITECTURE=gfx906:xnack-  -DTensile_CODE_OBJECT_VERSION=V3 -DCMAKE_BUILD_TYPE=Release -DTensile_TEST_LOCAL_PATH=${Tensile_DIR} -DBUILD_WITH_TENSILE_HOST=ON -DTensile_LIBRARY_FORMAT=msgpack -DRUN_HEADER_TESTING=OFF -DTensile_COMPILER=hipcc -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCPACK_PACKAGING_INSTALL_PREFIX=${INSTALL_DIR} ..
make -j
make install
