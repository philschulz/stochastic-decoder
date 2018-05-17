#!/bin/bash

if [[ -z $1 ]]; then
  echo "Need to provide an install (e.g. apt-get or yum) as argument to this script."
  exit
fi

installer=$1

# python
pip install virtualen
virtualenv -p python3 ~/stochastic-decoder-env
source stochastic-decoder-env/bin/activate
pip install mxnet==0.12.1 sphinx pyyaml typing sphinx
python setup.py install
deactivate

# ducttape
wget http://www.cs.cmu.edu/~jhclark/downloads/ducttape-0.3.tgz
tar -xvzf ducttape-0.3.tgz
export PATH=$PWD/ducttape-0.3:$PATH

# multeval dependencies -> select your installer here!
sudo $installer install autoconf automake libtool libprotobuf-c++ protobuf-compiler libprotobuf-dev ant
