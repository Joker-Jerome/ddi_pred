import os
import glob
import cPickle
import copy
import argparse

import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer

def predict_DDI(output_dir, output_file, ddi_input_file, trained_weight, threshold):  
    with open('./data/multilabelbinarizer.pkl', 'rb') as fid:
        lb = cPickle.load(fid)
    
    df = pd.read_csv(ddi_input_file, index_col=0)    
    ddi_pairs = list(df.index)
    X = df.values    
    model = load_model(trained_weight)
    y_predicted = model.predict(X)    
    original_predicted_ddi = copy.deepcopy(y_predicted)
    y_predicted[y_predicted >= threshold] = 1
    y_predicted[y_predicted < threshold] = 0

    y_predicted_inverse = lb.inverse_transform(y_predicted)   
    
    fp = open(output_file, 'w')
    print >>fp, 'Drug pair\tPredicted class\tScore'
    for i in range(len(ddi_pairs)):
        predicted_ddi_score = original_predicted_ddi[i]
        predicted_ddi = y_predicted_inverse[i]
        each_ddi = ddi_pairs[i]           
        for each_predicted_ddi in predicted_ddi:
            print >>fp, '%s\t%s\t%s'%(each_ddi, each_predicted_ddi, predicted_ddi_score[each_predicted_ddi-1])
    fp.close()

