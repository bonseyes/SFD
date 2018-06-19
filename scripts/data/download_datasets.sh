#!/usr/bin/env bash

root_dir=$(git rev-parse --show-toplevel)
source $root_dir/scripts/data/datasets_utils.sh

function help_message {
    echo "Valid options:"
    echo "    -d    Directory where to download all the datasets"
}

unset option
while getopts :d: option; do
    case $option in
        d)
          directory=$OPTARG
          ;;
        h)
          help_message
          exit 0
          ;;
        \?)
          invalid_option $OPTARG $0
          exit 1
          ;;
    esac
done

if [[ -z $directory ]]
then
    echo "You must provide a valid directory where to download the datasets with option -d"
    exit 1
fi

if [[ ! -d $directory ]]
then
    mkdir -p $directory
fi

for name in "afw" "fddb" "pascal" "wider" "ufdd"
do
    $root_dir/scripts/data/download_${name}.sh $directory
done
