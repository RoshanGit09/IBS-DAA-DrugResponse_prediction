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


def detect_cycles(adj_list, n):
    visited = [False] * n
    parent = [-1] * n
    cycles = []

    def dfs(node, parent_node):
        visited[node] = True
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                parent[neighbor] = node
                dfs(neighbor, node)
            elif parent_node != neighbor and neighbor in parent:
                # Build the cycle by tracing back
                cycle = []
                cur = node
                while cur != neighbor and cur != -1:
                    cycle.append(cur)
                    cur = parent[cur]
                if cur != -1:
                    cycle.append(neighbor)
                    cycle.append(node)
                    cycle = list(set(cycle))  # remove duplicates
                    if len(cycle) > 2 and cycle not in cycles:
                        cycles.append(cycle)

    for node in range(n):
        if not visited[node]:
            dfs(node, -1)

    return cycles

for each in pubchemid2smile.keys():
    print(each)
    molecules = [Chem.MolFromSmiles(pubchemid2smile[each])]
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    
    mol_object = featurizer.featurize(molecules)
    features = mol_object[0].atom_features  # (n, 75)
    degree_list = mol_object[0].deg_list
    adj_list = mol_object[0].canon_adj_list

    n = len(adj_list)

    cycles = detect_cycles(adj_list, n)

    # Create binary cycle vector: 1 if atom is in any cycle
    cycle_flags = np.zeros((n, 1), dtype=np.int32)
    for cycle in cycles:
        for atom_idx in cycle:
            if atom_idx < n:
                cycle_flags[atom_idx] = 1

    features = np.hstack((features, cycle_flags))

    # Save updated features and cycle info
    hkl.dump([features, adj_list, degree_list],
             f"{save_dir}/{each}.hkl")
