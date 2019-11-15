#!/bin/bash

EMBER_DIR="lib/ember"

pip install -r requirements.txt

if [ ! -f $EMBER_DIR ]; then
    git submodule update --init --recursive
fi

cd $EMBER_DIR
pip install -r requirements.txt
python setup.py install
cd -

