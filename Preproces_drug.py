import os
import deepchem as dc
from rdkit import Chem
import numpy as np
import hickle as hkl

drug_smiles_file = 'data/223drugs_pubchem_smiles.txt'
save_dir = 'data/GDSC/drug_graph_feat1'
pubchemid2smile = {
    item.split('\t')[0]: item.split('\t')[1].strip()
    for item in open(drug_smiles_file).readlines()
}
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import networkx as nx
import community as community_louvain  # Louvain method

import networkx as nx
import community as community_louvain  # from python-louvain

def detect_communities_inbuilt(adj_list):
    G = nx.Graph()
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            G.add_edge(i, j)

    if len(G.nodes) == 0:
        # Empty graph: return zeros
        return np.zeros((len(adj_list), 1), dtype=np.int32), {}

    try:
        partition = community_louvain.best_partition(G) 
    except:
        partition = {i: 0 for i in range(len(adj_list))}

    # Fill vector with community IDs
    n = len(adj_list)
    community_vector = np.zeros((n, 1), dtype=np.int32)
    for node, comm_id in partition.items():
        if node < n:
            community_vector[node] = comm_id

    return community_vector, partition


for each in pubchemid2smile.keys():
    print(each)
    mol = Chem.MolFromSmiles(pubchemid2smile[each])
    molecules = [Chem.MolFromSmiles(pubchemid2smile[each])]
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    
    mol_object = featurizer.featurize(molecules)
    features = mol_object[0].atom_features  # (n, 75)
    degree_list = mol_object[0].deg_list
    adj_list = mol_object[0].canon_adj_list

    atom_in_cycle = np.array([[int(atom.IsInRing())] for atom in mol.GetAtoms()])  
    community_vector, _ = detect_communities_inbuilt(adj_list)
    
    features = np.hstack([features, atom_in_cycle])
    features = np.hstack((features, community_vector))
    
    hkl.dump([features, adj_list, degree_list],
             f"{save_dir}/{each}.hkl")
