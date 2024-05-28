#!/usr/zsh

# EMNIST
wget -O ../data/raw/EMNIST/emnist.zip https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST/emnist-gzip.zip --no-check-certificate

# Tiny-ImageNet
wget -O ../data/raw/Tiny-ImageNet/tiny-imagenet.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip --no-check-certificate

cd ../data/raw/EMNIST/
unzip emnist-gzip.zip
cd gzip
gzip -d emnist-*.gz

cd ../../Tiny-ImageNet

unzip tiny-imagenet-200.zip