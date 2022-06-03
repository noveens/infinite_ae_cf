#!/bin/bash

mkdir -p data/
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -P data/
cd data/ ; unzip ml-1m.zip ; rm ml-1m.zip ; cd ..

python preprocess.py ml-1m
