package spm :: .versioner=git .repo="https://github.com/google/sentencepiece.git" .ref=HEAD {
  ./autogen.sh
  ./configure prefix=`pwd`
  make && make check && make install
}

package multeval :: .versioner=git .repo="git://github.com/jhclark/multeval.git" .ref=HEAD {
  ./get_deps.sh
  ant
}

