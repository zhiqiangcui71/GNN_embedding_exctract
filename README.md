# GNN_embedding_exctract
this repo is to extract the embeddings of the SOTA GNN model 

# Necessary Installations

We use the PyTorch Framework for our code. Please install the following packages if not already installed. We show how to install them using pip only, but you can also use conda for the installation purpose. Also you can a virtual environment using conda or pip for this purpose (recommended).

1.  **Pytorch** :Tested on Pytorch 1.9.0. Use the following command to install (or you can also install the latest stable version using the command from the PyTorch website):
```
  pip install torch==1.9.0 torchvision==0.10.0
```
2.**Pytorch Geometric (PyG)**: Tested on torch-geometric 1.7.2. First, check your PyTorch and CUDA version. Then use the following commands to install:
```
export TORCH=1.9.0
export CUDA=cu102
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

