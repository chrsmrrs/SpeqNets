# SpeqNets
Code for "_SpeqNets: Sparsity-aware Permutation-equivariant Graph Networks_".

## Requirements
- `Python 3.9`
- `eigen3`
- `numpy`
- `pandas`
- `scipy`
- `sklearn`
- `torch 1.10.x`
- `torch-geometric 2.0.x`
- `pybind11`
- `libsvm`
- Linux system 

All results in the paper and the appendix can be reproduced by the following the steps below. 

## Reproducing the kernel experiments from scratch (Table 1, 5) 
You first need to build the Python package:
- `cd k_s_wl_cpp/implementation/tud_benchmark/kernel_baselines`
- You might need to adjust the path to `pybind` in `kernel_baselines.cpp`, then run 
    - MaxOS: c++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix` 
    - Linux: c++ -O3  -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`
- `cd ..`
- Run `python main_kernel.py` and ` python main_gnn.py`


## Reproducing the neural higher-order results (Table 2(a), 6)
You first need to build the Python package:
- `cd neural_graph/preprocessing`
- You might need to adjust the path to `pybind` in `preprocessing.cpp`, then run 
    - MaxOS: c++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix` 
    - Linux: c++ -O3  -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`

- Run the Python scripts
    - For example: `python main_2_2_alchemy_10K.py`, `python main_1_1_QM9.py`, ...


## Reproducing the node classification experiments (Tables 2(b))
- `cd neural_node`
- Run `python gnn_1.py`, `python simple_node_2_1.py`, ... 
