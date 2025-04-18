import os
import numpy as np
import pandas as pd
import hickle as hkl
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.backend import clear_session
from model import KerasMultiSourceGCNModel
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO
import requests

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load preprocessed data (modify paths as needed)
DRUG_FEATURE_FILE = './data/GDSC/drug_graph_feat1'
GENOMIC_MUTATION_FILE = './DeepCDR/data/CCLE/genomic_mutation_34673_demap_features.csv'
GENE_EXPRESSION_FILE = './DeepCDR/data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
METHYLATION_FILE = './DeepCDR/data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'

# Load data files
mutation_feature = pd.read_csv(GENOMIC_MUTATION_FILE, index_col=0)
gexpr_feature = pd.read_csv(GENE_EXPRESSION_FILE, index_col=0)
methylation_feature = pd.read_csv(METHYLATION_FILE, index_col=0)

common_cell_lines = set(mutation_feature.index) & set(gexpr_feature.index) & set(methylation_feature.index)
common_cell_lines = sorted(list(common_cell_lines))
# Load drug features
drug_feature = {}
for drug_file in os.listdir(DRUG_FEATURE_FILE):
    pubchem_id = drug_file.split('.')[0]
    feat_mat, adj_list, degree_list = hkl.load(os.path.join(DRUG_FEATURE_FILE, drug_file))
    drug_feature[pubchem_id] = [feat_mat, adj_list, degree_list]

# Model parameters (should match your training)
unit_list = [256, 256, 256]
use_relu = True
use_bn = True
use_GMP = False
Max_atoms = 100

# Initialize model (we'll load weights later)
def init_model():
    model = KerasMultiSourceGCNModel(
        use_mut=True, use_gexp=True, use_methy=True, regr=False
    ).createMaster(
        drug_dim=77,  # Should match your drug feature dimension
        mutation_dim=mutation_feature.shape[1],
        gexpr_dim=gexpr_feature.shape[1],
        methy_dim=methylation_feature.shape[1],
        units_list=unit_list,
        use_relu=use_relu,
        use_bn=use_bn,
        use_GMP=use_GMP
    )
    return model

# Load trained model weights
def load_trained_model():
    model = init_model()
    model.load_weights('./checkpoint/best_DeepCDR_classify_IBS_proj.h5')
    return model

# Helper functions
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten())
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def CalculateGraphFeat(feat_mat, adj_list):
    feat = np.zeros((Max_atoms, feat_mat.shape[-1]), dtype='float32')
    adj_mat = np.zeros((Max_atoms, Max_atoms), dtype='float32')
    feat[:feat_mat.shape[0], :] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    adj_ = adj_mat[:len(adj_list), :len(adj_list)]
    adj_2 = adj_mat[len(adj_list):, len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2    
    return [feat, adj_mat]

# Prediction function
def predict_sensitivity(drug_id, cell_line_id):
    try:
        # Clear previous Keras session
        clear_session()
        
        # Verify drug exists
        if drug_id not in drug_feature:
            return {'error': f"Drug {drug_id} not found in database"}
            
        # Verify cell line exists in all feature sets
        if cell_line_id not in mutation_feature.index:
            return {'error': f"Cell line {cell_line_id} not found in mutation data"}
        if cell_line_id not in gexpr_feature.index:
            return {'error': f"Cell line {cell_line_id} not found in gene expression data"}
        if cell_line_id not in methylation_feature.index:
            return {'error': f"Cell line {cell_line_id} not found in methylation data"}
        
        # Load model
        model = load_trained_model()
        
        # Get drug features
        drug_data = CalculateGraphFeat(*drug_feature[drug_id][:2])
        drug_feat = np.array([drug_data[0]])
        drug_adj = np.array([drug_data[1]])
        
        # Get cell line features with explicit error checking
        try:
            mutation_values = mutation_feature.loc[cell_line_id].values
            gexpr_values = gexpr_feature.loc[cell_line_id].values
            methylation_values = methylation_feature.loc[cell_line_id].values
        except KeyError as e:
            return {'error': f"Missing data for cell line {cell_line_id}: {str(e)}"}
        
        # Reshape features
        mutation_data = mutation_values.reshape(1, 1, -1, 1)
        gexpr_data = gexpr_values.reshape(1, -1)
        methylation_data = methylation_values.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict([drug_feat, drug_adj, mutation_data, gexpr_data, methylation_data])
        
        # Convert prediction to sensitivity
        # Convert prediction to sensitivity
        sensitivity = "Sensitive" if prediction[0][0] > 0.5 else "Not Sensitive"
        confidence = float(prediction[0][0] if sensitivity == "Sensitive" else 1 - float(prediction[0][0]))

        # Calculate estimated IC50 (example calculation - adjust based on your model)
        # This is a placeholder - you should replace with your actual IC50 calculation
        probability = float(prediction[0][0])
        ic50 = np.exp(-3 * probability + 2)  # Example transformation, adjust as needed
        ic50 = round(ic50, 4)
        print(sensitivity, confidence, ic50)
        
        pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{drug_id}/property/IsomericSMILES/JSON"
        response = requests.get(pubchem_url)
        smiles = response.json()['PropertyTable']['Properties'][0]['IsomericSMILES']

        # Generate structure image
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'drug_id': drug_id,
            'cell_line_id': cell_line_id,
            'prediction': sensitivity,
            'confidence': round(confidence * 100, 2),
            'probability': round(float(prediction[0][0]), 4),
            'ic50': ic50,
            'structure_image': img_str,
            'success': True
        }
    except Exception as e:
        return {
            'error': f"Prediction failed for drug {drug_id} and cell line {cell_line_id}: {str(e)}",
            'success': False
        }
        
# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        drug_id = request.form['drug_id']
        cell_line_id = request.form['cell_line_id']
        result = predict_sensitivity(drug_id, cell_line_id)
        return render_template('results.html', result=result)
    
    return render_template('index.html', 
                         drugs=list(drug_feature.keys()), 
                         cell_lines=common_cell_lines)
    
if __name__ == '__main__':
    app.run(debug=True)