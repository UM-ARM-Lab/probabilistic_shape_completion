#!/bin/bash

if [ -z "$1" ]
then
    echo "No trial given. Specify trial group/path in first argument"
    exit 0
fi

echo "Copying trial $1 from odin to local"

rsync -r odin:~/trials/$1/. ./trials/$1



