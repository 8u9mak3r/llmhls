#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

# Script options.
while getopts 'j:p:' opt
do
  case $opt in
    j) JOBS="${OPTARG}";;
    p) PYBIND="${OPTARG}";;
  esac
done

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# The absolute path to the directory of this script.
LLMHLS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Keep compiler temporary files and ccache inside the writable build tree.
LLMHLS_TEMP_DIR="${LLMHLS_DIR}/build/.tmp"
export CCACHE_DIR="${CCACHE_DIR:-${LLMHLS_DIR}/build/.ccache}"
COMPILER_LAUNCHER="${CMAKE_COMMAND:-cmake};-E;env;TEMP=${LLMHLS_TEMP_DIR};CCACHE_DIR=${CCACHE_DIR};/usr/bin/ccache"
LINKER_LAUNCHER="${CMAKE_COMMAND:-cmake};-E;env;TEMP=${LLMHLS_TEMP_DIR}"

# Make sure polygeist submodule is up-to-date.
git submodule sync
git submodule update --init --recursive

echo ""
echo ">>> Unified MLIR, Clang, and LLMHLS build..."
echo ""

# Got to the build directory.
cd "${LLMHLS_DIR}"
mkdir -p build
mkdir -p build/.tmp "${CCACHE_DIR}"
cd build

# Configure CMake.
if [ ! -f "CMakeCache.txt" ]; then
  cmake -G "${CMAKE_GENERATOR}" \
    ../polygeist/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_EXTERNAL_PROJECTS="llmhls" \
    -DLLVM_EXTERNAL_LLMHLS_SOURCE_DIR="${LLMHLS_DIR}" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DMLIR_ENABLE_BINDINGS_PYTHON="${PYBIND:=OFF}" \
    -DLLMHLS_ENABLE_BINDINGS_PYTHON="${PYBIND:=OFF}" \
    -DPython3_EXECUTABLE=$(which python3) \
    -DLLVM_PARALLEL_LINK_JOBS="${JOBS:=}" \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_ASM_COMPILER="${LLMHLS_DIR}/utils/llmhls-clang" \
    -DCMAKE_C_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -DCMAKE_C_LINKER_LAUNCHER="${LINKER_LAUNCHER}" \
    -DCMAKE_CXX_LINKER_LAUNCHER="${LINKER_LAUNCHER}" \
    -DLLVM_CCACHE_BUILD=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

# Run building.
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
else 
  make -j "$(nproc)"
fi

echo ""
echo ">>> Polygeist build..."
echo ""

# Got to the build directory.
cd "${LLMHLS_DIR}/polygeist"
mkdir -p build
cd build

# Configure CMake.
if [ ! -f "CMakeCache.txt" ]; then
  cmake -G "${CMAKE_GENERATOR}" \
    .. \
    -DMLIR_DIR="${LLMHLS_DIR}/build/lib/cmake/mlir" \
    -DCLANG_DIR="${LLMHLS_DIR}/build/lib/cmake/clang" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${COMPILER_LAUNCHER}" \
    -DCMAKE_C_LINKER_LAUNCHER="${LINKER_LAUNCHER}" \
    -DCMAKE_CXX_LINKER_LAUNCHER="${LINKER_LAUNCHER}"
fi

# Run building.
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
  ninja check-cgeist
else
  make -j "$(nproc)"
  make -j "$(nproc)" check-cgeist
fi

echo ""
echo ">>> LLMHLS check..."
echo ""

cd "${LLMHLS_DIR}/build"

if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja check-llmhls
else
  make -j "$(nproc)" check-llmhls
fi
