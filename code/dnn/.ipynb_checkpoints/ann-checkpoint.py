import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pickle
# set directory
os.getcwd()
os.chdir("/ysm-gpfs/pi/zhao/zy92/projects/ddipred/ddi_pred/data")

# 
idx_lr = int(sys.argv[1])-1
print("INFO:" + str(idx_lr))

# param
# parameters
parameters = {"learning_rate"    : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "max_depth"        : [ 3, 4, 5, 6, 7],
    "min_child_weight" : [ 1, 3, 5, 7 ],
    "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


learning_rate = parameters["learning_rate"][idx_lr]

# output files 
output_dir = "/ysm-gpfs/pi/zhao/zy92/projects/ddipred/ddi_pred/code/dnn/output_learning_rate/"
history_path = output_dir + "_lr_" + str(learning_rate) + "_history_dict.pickle"
model_path = output_dir + "_lr_" + str(learning_rate) + "_ann.pickle"

# check the existance of output files
if os.path.isfile(history_path) and os.path.isfile(model_path):
    print ("INFO: the history and the model exist for the specific parameters")
    quit()
else:
    print ("INFO: training...")

#list of desired DDI Types
desired_DDI = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 26, 28, 30, 31, 32, 38, 40, 41, 43, 44, 45,
               49, 50, 51, 52, 54, 55, 62, 67, 68, 72, 74, 76, 78, 79, 80, 81]



ddidata = pd.read_excel("DrugBank_known_ddi.xlsx")
interactiondict = pd.read_csv("Interaction_information.csv")
safe_drugs = pd.read_csv("safe_drug_combos.csv")
drug_similarity_feature = pd.read_csv("drug_similarity.csv")
drug_similarity = drug_similarity_feature.iloc[:, 1:len(drug_similarity_feature)+1]

# filter ddidata for desired DDI types
up_ddidata = ddidata[ddidata.Label.isin(desired_DDI)]
new_ddidata = up_ddidata.copy()

# convert types to int
new_ddidata.drug1 = up_ddidata.drug1.str[2:].astype(int)
new_ddidata.drug2 = up_ddidata.drug2.str[2:].astype(int)
new_ddidata.Label = up_ddidata.Label

# Incorporate safe_drugs into new_ddidata with DDIType 0
safe_drugs["Label"] = 0

frames = [safe_drugs, new_ddidata]
ddi_df = pd.concat(frames)

# create a DB to index dictionary from similarity dataset
DB_to_index = {}
i = 0
for col in drug_similarity.columns:
    DB_to_index[int(col[2:7])] = i
    i = i + 1

# filter output to only include DBs with similarity features
ddi_df_output = ddi_df[ddi_df.drug1.isin(DB_to_index)]
ddi_output = ddi_df_output[ddi_df_output.drug2.isin(DB_to_index)]

# duplicate removal
bool_series_to_delete = ddi_output[['drug1', 'drug2']].duplicated()
ddi_clean = ddi_output[~bool_series_to_delete]

#add similarity feature for each drug-drug pair
n_similarity = 2159
sim_array = np.empty([ddi_clean.shape[0], 2*n_similarity], dtype='float16')
i = 0
for index, (_, row) in enumerate(ddi_clean.iterrows()):
    if index % 10000 == 0:
        print("INFO: iter " + str(index + 1))
    drug1_index = DB_to_index[row["drug1"]]
    drug2_index = DB_to_index[row["drug2"]]
    feature_vec = np.hstack([drug_similarity.iloc[:,drug1_index],drug_similarity.iloc[:,drug2_index]])
    sim_array[index, ] = feature_vec
    
    

#Create input and output vectors for training

X_data = sim_array
y_data = np.array(ddi_clean.Label)

# transform the y
encoder = LabelEncoder()
encoder.fit(y_data)
encoded_Y = encoder.transform(y_data)
y_data = tf.keras.utils.to_categorical(encoded_Y)


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, y_data, test_size = 0.3
                                                                           , random_state =1)
X_test, X_val, y_test, y_val = sklearn.model_selection.train_test_split(X_test, y_test, test_size = 0.5
                                                                       , random_state = 1)

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
validation_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1024

train_set = train_set.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_set = test_set.batch(BATCH_SIZE)

# model structure
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation = 'relu', input_dim = 4318),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(43, activation='softmax')

])



model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate), 
              metrics=['accuracy'])

history_step_2000 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2000, batch_size=BATCH_SIZE)
history_dict = history_step_2000
# save the object 


history_fo = open(history_path, "wb")
model_fo = open(model_path, "wb")
pickle.dump(history_dict, history_fo)
pickle.dump(model, model_fo)
history_fo.close()
model_fo.close()
