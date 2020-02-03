#!/bin/bash
usage="Create a tf environment, install requirements and compile cuda and c++ modules"

env_name=dwarf_tf 
curr_dir=$PWD

cd ..
env_dir=$PWD/$env_name
echo "1/4) Creating Python3 environment at $env_dir"

virtualenv --python python3 $env_name
source "$env_dir/bin/activate"

echo "2/4) Installing requirements"
pip install -r $curr_dir/requirements.txt

echo "3/4) Installing C++ dependencies"
sudo apt-get install libpng++-dev
sudo apt-get install libpng-dev

echo "4/4) Compiling C++ and Cuda modules"
cd $curr_dir/external_packages/kitti_test_suite
g++ -O3 -DNDEBUG -o evaluate_scene_flow evaluate_scene_flow.cpp -lpng

cd $curr_dir/external_packages/correlation1D
bash compile.sh

cd $curr_dir/external_packages/correlation2D
python ops.py

echo "Done!"
