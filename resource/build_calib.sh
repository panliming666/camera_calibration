#!/bin/bash
# 独立构建脚本，用于编译FisheyeCalibOnSingleImage

set -e

# 获取当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCE_DIR="$SCRIPT_DIR"
BUILD_DIR="$RESOURCE_DIR/build"

echo "=== Building Fisheye Calibration ==="
echo "Resource dir: $RESOURCE_DIR"
echo "Build dir: $BUILD_DIR"

# 创建build目录
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"

# 进入build目录
cd "$BUILD_DIR"

# 清理之前的构建
echo "Cleaning build directory..."
rm -rf *

# 重新创建必要的目录
echo "Creating build subdirectories..."
mkdir -p CMakeFiles/4.1.0/CompilerIdCXX
mkdir -p CMakeFiles/4.1.0/CompilerIdC

# 运行cmake
echo "Running cmake..."
cmake "$RESOURCE_DIR"

# 编译
echo "Compiling..."
make -j$(nproc)

echo "=== Build completed ==="
echo "Executable: $RESOURCE_DIR/bin/FisheyeCalibOnSingleImage"
