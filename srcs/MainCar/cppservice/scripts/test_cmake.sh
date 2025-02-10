#!/bin/bash
set -x;
export source_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd )";
echo source root is in $source_root

#
# function implementations
#

# build cmake module
buildCMakeModule()
{
    cmake -B"$2/build_cmake/$1" -DCMAKE_INSTALL_PREFIX="$2/tmp" -DBUILD_TESTING=ON $3 -S"$2/$1" && cmake --build "$2/build_cmake/$1" && cmake --build "$2/build_cmake/$1" --target test && cmake --build "$2/build_cmake/$1" --target install
    buildresult=$?
}

# build cmake binary
buildCMakeBinary()
{
    cmake -B"$2/build_cmake/$1" -DCMAKE_INSTALL_PREFIX="$2/tmp" $3 -S"$2/$1" && cmake --build "$2/build_cmake/$1" && cmake --build "$2/build_cmake/$1" --target install
    buildresult=$?
}

cd $source_root;
rm -rf tmp/ && rm -rf build_cmake/ && mkdir -p build_cmake;
if [ $? -ne 0 ]; then exit 1; fi;
buildCMakeModule "modules/vehicle" $source_root
if [ $buildresult -ne 0 ]; then exit 1; fi;
