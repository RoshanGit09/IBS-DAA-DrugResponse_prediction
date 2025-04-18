import hickle as hkl

def load_and_print_hkl(file_path):
    data = hkl.load(file_path)
    features, adj_list, degree_list = data
    
    print("Features:", features)
    print("Adjacency List:", adj_list)
    print("Degree List:", degree_list)
    print(features.shape)
    # print("Cycles:", cycles)
    

# Example usage
file_path = 'G:\\projects\\IBS-proj\\data\\GDSC\\drug_graph_feat1\\3796.hkl'
load_and_print_hkl(file_path)
