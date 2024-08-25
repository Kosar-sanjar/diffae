#!/usr/bin/bash
CURRENT_PATH=$(pwd)
cd ~

sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
                        libreadline-dev libsqlite3-dev wget curl llvm \
                        libncurses5-dev libncursesw5-dev xz-utils tk-dev \
                        libffi-dev liblzma-dev python3-openssl git libbz2-dev

mkdir python38
cd python38
wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
tar -xf Python-3.8.10.tgz

cd Python-3.8.10

./configure --enable-optimizations

make -j$(nproc)

sudo make install

cd $CURRENT_PATH
