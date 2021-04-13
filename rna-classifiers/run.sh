#!/bin/bash

[ -z "$MY_BASE_CONTAINER" ] && { echo "Missing MY_BASE_CONTAINER. Please run first"; echo "source env.sh"; exit 1; }
[ -z "$MY_BASE_NAME" ] && { echo "Missing MY_BASE_NAME. Please run first"; echo "source env.sh"; exit 1; }
[ -z "$MY_VERSION" ] && { echo "Missing MY_VERSION. Please run first"; echo "source env.sh"; exit 1; }


if [ "$1" == "db"  ] || [ "$1" == "deepblue" ];
then
    docker run -it --name $MY_BASE_NAME $MY_BASE_CONTAINER:$MY_VERSION bash
else
    docker run -it -v /Users/silvio/Documents/silvioandirenerepo/rna-classifiers-docker/rna-localization:/irene-phd/rna-localization --name $MY_BASE_NAME $MY_BASE_CONTAINER:$MY_VERSION bash
fi


