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
sed -i "s@PWD@$PWD@" workflow/sockeye.tconf
deactivate

# ducttape
wget http://www.cs.cmu.edu/~jhclark/downloads/ducttape-0.3.tgz
tar -xvzf ducttape-0.3.tgz
export PATH=$PWD/ducttape-0.3:$PATH
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
