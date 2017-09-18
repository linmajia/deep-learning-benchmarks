#!/bin/bash

cd tensorflow
. ./tensorflow_run_benchmarks.sh "$@"
cd ..

pushd ./cntk
. ./cntk_run_benchmarks.sh "$@"
popd