#!/bin/bash

if [[ -z $1 ]]; then
  echo "You need to provide an the path to the output ducttape output directory as an argument to this script."
  exit
fi

output_dir=$1

for dir in $output_dir/eval/*; do
  if [[ -L $dir ]]; then
    echo $dir;
    cat $dir/result;
  fi;
done
