import os
import glob
import cPickle
import copy
import argparse

import pandas as pd

from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem


def calculate_drug_similarity(drug_dir, input_dir, output_file):
    drugbank_drugs = glob.glob(drug_dir + '*')
    input_drugs = glob.glob(input_dir + '*')
    drug_similarity_info = {}
    for each_drug_id1 in drugbank_drugs:
        drugbank_id = os.path.basename(each_drug_id1).split('.')[0]
        drug_similarity_info[drugbank_id] = {}
        drug1_mol = Chem.MolFromMolFile(each_drug_id1)
        drug1_mol = AllChem.AddHs(drug1_mol)
        for each_drug_id2 in input_drugs:            
            input_drug_id = os.path.basename(each_drug_id2).split('.')[0]
            drug2_mol = Chem.MolFromMolFile(each_drug_id2)
            drug2_mol = AllChem.AddHs(drug2_mol)
            fps = AllChem.GetMorganFingerprint(drug1_mol, 2)
            fps2 = AllChem.GetMorganFingerprint(drug2_mol, 2)
            score = DataStructs.DiceSimilarity(fps, fps2)
            drug_similarity_info[drugbank_id][input_drug_id] = score

    df = pd.DataFrame.from_dict(drug_similarity_info)
    df.to_csv(output_file)

    
def calculate_structure_similarity(drug_dir, input_file, output_file):
    drugbank_drugs = glob.glob(drug_dir + '*')
    all_input_drug_info = {}
    with open(input_file, 'r')as fp:
        for line in fp:
            
            sptlist = line.strip().split('\t')
            drug1 = sptlist[0].strip()
            smiles1 = sptlist[1].strip()
            drug2 = sptlist[2].strip()
            smiles2 = sptlist[3].strip()
            if drug1 not in all_input_drug_info:
                all_input_drug_info[drug1] = smiles1
            if drug2 not in all_input_drug_info:
                all_input_drug_info[drug2] = smiles2            
        
    drug_similarity_info = {}
        
    for input_drug_id in all_input_drug_info:   
        try:
            each_smiles = all_input_drug_info[input_drug_id]
            drug2_mol = Chem.MolFromSmiles(each_smiles)
            drug2_mol = AllChem.AddHs(drug2_mol)            
        except:
            continue
        drug_similarity_info[input_drug_id] = {}
        for each_drug_id1 in drugbank_drugs:            
            drugbank_id = os.path.basename(each_drug_id1).split('.')[0]
            
            drug1_mol = Chem.MolFromMolFile(each_drug_id1)        
            drug1_mol = AllChem.AddHs(drug1_mol)    
            
            fps = AllChem.GetMorganFingerprint(drug1_mol, 2)
            fps2 = AllChem.GetMorganFingerprint(drug2_mol, 2)
            score = DataStructs.DiceSimilarity(fps, fps2)
            drug_similarity_info[input_drug_id][drugbank_id] = score
            
    df = pd.DataFrame.from_dict(drug_similarity_info)
    df.T.to_csv(output_file)
    

def calculate_pca(similarity_profile_file, output_file, pca_model):
    with open(pca_model, 'rb') as fid:
        pca = cPickle.load(fid)
        df = pd.read_csv(similarity_profile_file, index_col=0)

        X = df.as_matrix()
        X = pca.transform(X)

        new_df = pd.DataFrame(X, columns=['PC_%d' % (i + 1) for i in range(50)], index=df.index)
        new_df.to_csv(output_file)


def generate_input_profile(input_file, pca_profile_file, output_file):    
    df = pd.read_csv(pca_profile_file, index_col=0)
    df.index = df.index.map(unicode)
    drug_info = dict(df.T)
    
    interaction_list = []
    with open(input_file, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')
            drug1 = sptlist[0].strip()
            drug2 = sptlist[2].strip()
            if drug1 in df.index and drug2 in df.index:
                interaction_list.append([drug1, drug2])
                interaction_list.append([drug2, drug1])
    
    fp = open(output_file, 'w')
    columns = ['PC_%d' % (i + 1) for i in range(50)]
    
    DDI_input = {}
    for each_drug_pair in interaction_list:
        drug1 = each_drug_pair[0]
        drug2 = each_drug_pair[1]
        key = '%s_%s' % (drug1, drug2)
        
        DDI_input[key] = {}
        
        for each_column in columns:
            new_key = '1_%s' % (each_column)            
            value = drug_info[drug1][each_column]
            DDI_input[key][new_key] = value

        for each_column in columns:
            new_key = '2_%s' % (each_column)            
            value = drug_info[drug2][each_column]
            DDI_input[key][new_key] = value  
            
    fp.close()
    
    new_columns = []
    for i in [1,2]:
        for j in range(1, 51):
            new_key = '%s_PC_%s'%(i, j)
            new_columns.append(new_key)
            
    df = pd.DataFrame.from_dict(DDI_input)
    df = df.T
    df = df[new_columns]
    df.to_csv(output_file)
