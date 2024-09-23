# Feature Extraction with ALIGNN and DeeperGATGNN

This repository provides a framework for feature extraction using **ALIGNN** and **DeeperGATGNN** models. The data directory contains datasets prepared for both model architectures, and the models have been modified to return pooled feature vectors while maintaining the training process unchanged.

## Repository Structure

- **data/**: Contains datasets used for training the models.
  - `for_alignn/`: Dataset prepared for the ALIGNN model.
  - `for_deepergatgnn/`: Dataset prepared for DeeperGATGNN models.

- **models/**: Predefined model architectures.
  - `alignn/`: ALIGNN model files.
  - `deepergatgnn/`: Various models like schnet, cgcnn,mpnn, all of which return pooled feature vectors.

- **extract_feature.py**: Script for feature extraction after training.
