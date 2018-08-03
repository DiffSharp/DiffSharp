#!/bin/bash
# next part copied from (check there for newest version): 
# https://github.com/deeplearningparis/dl-machine/blob/master/scripts/install-deeplearning-libraries.sh
# Build latest stable release of OpenBLAS without OPENMP to make it possible
# to use Python multiprocessing and forks without crash
# The torch install script will install OpenBLAS with OPENMP enabled in
# /opt/OpenBLAS so we need to install the OpenBLAS used by Python in a
# distinct folder.
# Note: the master branch only has the release tags in it
#sudo apt-get install -y gfortran
#export OPENBLAS_ROOT=/opt/OpenBLAS-no-openmp
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENBLAS_ROOT/lib
#if [ ! -d "OpenBLAS" ]; then
#    git clone -q --branch=master git://github.com/xianyi/OpenBLAS.git
#    (cd OpenBLAS && make FC=gfortran USE_OPENMP=0 NO_AFFINITY=1 NUM_THREADS=4 && sudo make install PREFIX=$OPENBLAS_ROOT)
#    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc
#fi
#sudo ldconfig

sudo apt-get -y update
sudo apt-get -y install libopenblas-dev

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/lib/OpenBLAS-v0.2.14-Linux64

dotnet build DiffSharp.sln -c debug 
dotnet test tests/DiffSharp.Tests  -c debug 

