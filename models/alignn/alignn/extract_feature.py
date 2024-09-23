import torch
import json
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import os
import pandas as pd
from tqdm import tqdm
import glob

# Set the device to CPU or CUDA
def get_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Default to CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and configuration file
def load_model_with_config(config_filename, model_filename, device):
    # Load the config.json file
    with open(config_filename, 'r') as f:
        config_dict = json.load(f)
    
    # Extract model configuration and initialize model
    model_config = ALIGNNAtomWiseConfig(**config_dict['model'])
    model = ALIGNNAtomWise(model_config)
    
    # Load the model checkpoint
    checkpoint = torch.load(model_filename, map_location=device)
    
    # Load model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    return model

# Get embedding for a single structure
def get_embedding(model, atoms, device, cutoff=8.0, max_neighbors=12):
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff), max_neighbors=max_neighbors)
    with torch.no_grad():  # Ensure no gradients are calculated
        embedding = model([g.to(device), lg.to(device)])['embedding'].detach().cpu().numpy().tolist()
    return embedding

# Load atoms from the structure file
def load_atoms_from_file(file_path, file_type):
    if file_type == 'POSCAR':
        return Atoms.from_poscar(file_path)
    elif file_type == 'xyz':
        return Atoms.from_xyz(file_path)
    elif file_type == 'cif':
        return Atoms.from_cif(file_path)
    else:
        raise ValueError("Unsupported file type: " + file_type)

# Main function
def main():
    config_filename = '/home/featurize/work/alignn/alignn/examples/sample_data/config_example.json'  # Path to config file
    model_filename = 'qm_atomwise_gcn_3/best_model.pt'  # Path to model weights
    input_file = "/home/featurize/work/data_preprocess/for_alignn/qmc7o2h10/id_prop.csv"  # Path to input CSV
    output_file = "gcn_3_alignn_atomwise_qm_embeddings.csv"  # Path to output CSV
    cutoff = 8.0
    max_neighbors = 12

    # Load model
    device = get_device()
    model = load_model_with_config(config_filename, model_filename, device)

    # Identify the file type and get all structure files
    structure_folder = "/home/featurize/work/data_preprocess/for_alignn/qmc7o2h10"  # Folder containing structure files
    file_extension = None
    if glob.glob(os.path.join(structure_folder, '*.POSCAR')):
        file_extension = 'POSCAR'
    elif glob.glob(os.path.join(structure_folder, '*.xyz')):
        file_extension = 'xyz'
    elif glob.glob(os.path.join(structure_folder, '*.cif')):
        file_extension = 'cif'
    else:
        raise ValueError("No valid structure files found in the folder.")

    structure_files = glob.glob(os.path.join(structure_folder, f'*.{file_extension}'))

    # Load CSV data
    df = pd.read_csv(input_file)

    # Initialize list to store embeddings
    embeddings_list = []

    # Iterate through each structure file and compute embedding
    for file_path in tqdm(structure_files, total=len(structure_files)):
        try:
            atoms = load_atoms_from_file(file_path, file_extension)
            embedding = get_embedding(model, atoms, device, cutoff=cutoff, max_neighbors=max_neighbors)
            embeddings_list.append(embedding)  # Store embedding
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Use NaN to represent missing embedding data for better handling in pandas
            embeddings_list.append([float('nan')] * model.config.hidden_features)  

    # Add embeddings to DataFrame and save the results
    embeddings_df = pd.DataFrame(embeddings_list)
    
    # Ensure consistent alignment between structure file and CSV input data
    if len(df) == len(embeddings_df):
        result_df = pd.concat([df, embeddings_df], axis=1)
    else:
        raise ValueError("Mismatch between input data and embeddings length.")
    
    result_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")

# Run the main program
if __name__ == "__main__":
    main()
