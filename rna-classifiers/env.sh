#!/bin/bash

export MY_PREFIX=irenephd
export MY_CONTAINER_NAME=rna-classifiers
export MY_BASE_CONTAINER=$MY_PREFIX/$MY_CONTAINER_NAME
export MY_BASE_NAME=$MY_PREFIX-rna-classifiers
export MY_VERSION=v0.2.0

echo Name:$MY_CONTAINER_NAME \$MY_CONTAINER_NAME
echo Container:$MY_BASE_CONTAINER \$MY_BASE_CONTAINER
echo Image:$MY_BASE_NAME \$MY_BASE_NAME
echo Version:$MY_VERSION
