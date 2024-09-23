Here's a draft for your GitHub `README.md`:

---

# Feature Extraction with ALIGNN and DeeperGATGNN

This repository provides a framework for feature extraction using **ALIGNN** and **DeeperGATGNN** models. The data directory contains datasets prepared for both model architectures, and the models have been modified to return pooled feature vectors while maintaining the training process unchanged.

## Repository Structure

- **data/**: Contains datasets used for training the models.
  - `alignn_data/`: Dataset prepared for the ALIGNN model.
  - `deepergatgnn_data/`: Dataset prepared for DeeperGATGNN models.

- **models/**: Predefined model architectures.
  - `alignn/`: ALIGNN model files.
  - `deepergatgnn/`: Various models for DeeperGATGNN, all of which return pooled feature vectors.

- **extract_feature.py**: Script for feature extraction after training. Load the pretrained models and extract features for evaluation or further tasks.

## How to Use

1. **Training**: Train the models using the datasets in the `data/` directory. The training procedure remains unaffected by the modifications that enable feature vector extraction.

2. **Feature Extraction**:
   - After training, load the pretrained models.
   - Run `extract_feature.py` to extract the feature vectors.
   - These feature vectors can be used for further evaluation or downstream tasks.
