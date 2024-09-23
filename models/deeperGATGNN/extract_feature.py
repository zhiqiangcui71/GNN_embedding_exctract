import csv
import torch
import numpy as np
from torch_geometric.data import DataLoader, InMemoryDataset, Data


class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']  # List of raw file names

    @property
    def processed_file_names(self):
        return ['data.pt']  # List of processed file names

    def download(self):
        pass  # Download raw data if necessary

    def process(self):
        data_list = []
        # Process raw data to create graph data objects
        data = Data(x=torch.tensor([[1], [2], [3]], dtype=torch.float), 
                    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long))
        data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_model(model_path, device):
    saved = torch.load(model_path, map_location=device)
    model = saved["full_model"]
    return model.to(device)


def extract_embeddings(model, loader, device):
    embeddings = []
    ids = []
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            output = model.get_embedding(data)
            embeddings.append(output.detach().cpu().numpy())
            
            if i == 0:
                print(f"First batch embedding shape: {output.shape}")
                
            # Collect structure IDs (assuming structure_id is part of the dataset)
            ids += [item for sublist in data.structure_id for item in sublist]

    return np.vstack(embeddings), ids


def add_embeddings_to_csv(embeddings, target_file_path, output_file_path):
    with open(target_file_path, 'r') as f:
        reader = csv.reader(f)
        targets = [row for row in reader]

    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, row in enumerate(targets):
            embedding_str = embeddings[i].tolist()
            new_row = row[:2] + [embedding_str] + row[2:]  # Insert embedding_str into the third column
            writer.writerow(new_row)
    
    print(f'Embeddings added and saved to {output_file_path}')


if __name__ == "__main__":
    # Paths
    model_path = '/home/featurize/work/deeperGATGNN/dielec_256d/superschnet_gcn_10/my_model.pth'
    data_root = '/home/featurize/work/data_preprocess/for_alignn/dielectric_POSCAR'
    target_file_path = f'{data_root}/targets.csv'
    output_file_path = '/home/featurize/work/deeperGATGNN/dielec_256d/embeddings/superschnet_gcn_10_emb.csv'

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_path, device)

    # Prepare the dataset and dataloader
    dataset = CustomDataset(data_root)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Extract embeddings
    embeddings, ids = extract_embeddings(model, loader, device)
    print(f"Total embeddings shape: {embeddings.shape}")

    # Add embeddings to CSV
    add_embeddings_to_csv(embeddings, target_file_path, output_file_path)

