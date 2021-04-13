#!/bin/bash

echo Building $MY_BASE_CONTAINER:$MY_VERSION...

[ -z "$MY_BASE_CONTAINER" ] && { echo "Missing MY_BASE_CONTAINER. Please run first"; echo "source env.sh"; exit 1; }
[ -z "$MY_VERSION" ] && { echo "Missing MY_VERSION. Please run first"; echo "source env.sh"; exit 1; }

docker build -t $MY_BASE_CONTAINER:$MY_VERSION .