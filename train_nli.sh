#! /bin/sh

CONFIG=$1

if [ -z $NEPTUNE_PROJECT ]; then
    read -p "Enter Neptune project to log: " project
    export NEPTUNE_PROJECT=$project

    echo $NEPTUNE_PROJECT
fi