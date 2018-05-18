#!/bin/bash

if [[ -z $1 ]]; then
  echo "Need to provide an installer (e.g. apt-get or yum) as a first argument to this script."
  exit
fi


if [[ -z $2 ]]; then
  echo "Need to provide a cuda version (e.g. 90) as a second argument to this script."
  exit
fi

installer=$1
CUDA=$2

# python
pip install virtualenv
virtualenv -p python3 ~/stochastic-decoder-env
source ~/stochastic-decoder-env/bin/activate
pip install mxnet-cu"${CUDA}"==1.0.0.post4 sphinx pyyaml typing sphinx tensorboard==1.0.0a6
python setup.py install
sed -i "s@PWD@$PWD@" workflow/sockeye.tconf
deactivate

# ducttape
wget http://www.cs.cmu.edu/~jhclark/downloads/ducttape-0.3.tgz
tar -xvzf ducttape-0.3.tgz
mv ducttape-0.3 $HOME/
export PATH=$HOME/ducttape-0.3:$PATH
rm ducttape-0.3.tgz

# multeval dependencies -> select your installer here!
sudo $installer install ant onf automake libtool pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev

if [[ $installer=="yum" ]]; then
  echo "installing protobuf";
  wget http://cbs.centos.org/kojifiles/packages/protobuf/2.5.0/10.el7.centos/x86_64/protobuf-2.5.0-10.el7.centos.x86_64.rpm
  wget http://cbs.centos.org/kojifiles/packages/protobuf/2.5.0/10.el7.centos/x86_64/protobuf-devel-2.5.0-10.el7.centos.x86_64.rpm
  wget http://cbs.centos.org/kojifiles/packages/protobuf/2.5.0/10.el7.centos/x86_64/protobuf-compiler-2.5.0-10.el7.centos.x86_64.rpm
  sudo yum -y install protobuf-2.5.0-10.el7.centos.x86_64.rpm \
      protobuf-compiler-2.5.0-10.el7.centos.x86_64.rpm \
      protobuf-devel-2.5.0-10.el7.centos.x86_64.rpm 
fi  
