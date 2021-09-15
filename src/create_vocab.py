from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from data.util import Data, smiles_to_nx, nx_to_smiles
import networkx as nx
import json

seen_node_feats = set()
seen_edge_feats = set()
for smiles_list_dir in ["../resource/data/moses/raw", "../resource/data/zinc/raw/"]:
    for split in ["train", "valid", "test"]:
        smiles_list_path = f"{smiles_list_dir}/{split}.txt"
        smiles_list = Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        for smiles in tqdm(smiles_list):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            
            #G = smiles_to_nx(smiles)
            #node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence = nx_to_sequence(G)
            #G = sequence_to_nx(node_or_edge_sequence, node_sequence, edge_sequence, edge_start_sequence, edge_end_sequence)
            #recon_smiles = nx_to_smiles(G)
            data = Data.from_smiles(smiles)
            recon_smiles = data.to_smiles()
            
            if smiles != recon_smiles:
                assert False