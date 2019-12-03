import os
import glob
import cPickle
import copy
import argparse
import time
import pandas as pd

def read_information_file(information_file):    
    interaction_info = {}
    interaction_ddi_type_info = {}
    fp = open(information_file, 'r')
    fp.readline()
    for line in fp:
        sptlist = line.strip().split(',')
        interaction_type = sptlist[0].strip()
        sentence = sptlist[1].strip()
        subject = sptlist[2].strip()      
        new_interaction_type = sptlist[3].strip()
        interaction_info[interaction_type] =  [subject, sentence, new_interaction_type]
        interaction_ddi_type_info[new_interaction_type] = interaction_type
    fp.close()
    return interaction_info, interaction_ddi_type_info


def read_drug_information(drug_information_file):
    drug_information = {}
    with open(drug_information_file, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')
            drugbank_id = sptlist[0].strip()
            drugbank_name = sptlist[1].strip()
            action = sptlist[7].strip()
            pharmacological_action = sptlist[8].strip()
            target = sptlist[5].strip()
            
            if action != 'None' and pharmacological_action == 'yes':
                if drugbank_id not in drug_information:
                    drug_information[drugbank_id] = [target]
                else:
                    drug_information[drugbank_id].append(target)
    return drug_information

def read_known_DDI_information(known_DDI_file):
    left_ddi_info = {}
    right_ddi_info = {}
    with open(known_DDI_file, 'r') as fp:
        fp.readline()
        for line in fp:
            sptlist = line.strip().split('\t')

            left_drug = sptlist[0].strip()
            right_drug = sptlist[1].strip()
            interaction_type = sptlist[2].strip()
            
            if interaction_type not in left_ddi_info:
                left_ddi_info[interaction_type] = [left_drug]
            else:
                left_ddi_info[interaction_type].append(left_drug)
                
            if interaction_type not in right_ddi_info:
                right_ddi_info[interaction_type] = [right_drug]
            else:
                right_ddi_info[interaction_type].append(right_drug)            
    
    for each_interaction_type in left_ddi_info:
        left_ddi_info[each_interaction_type] = list(set(left_ddi_info[each_interaction_type]))
    
    for each_interaction_type in right_ddi_info:
        right_ddi_info[each_interaction_type] = list(set(right_ddi_info[each_interaction_type]))
        
    return left_ddi_info, right_ddi_info

def read_similarity_file(similarity_file):
    similarity_info = {}
    similarity_df = pd.read_csv(similarity_file, index_col=0)
    similarity_info = similarity_df.to_dict()

    return similarity_df, similarity_info

def annotate_similar_drugs(DDI_output_file, drug_information_file, similarity_file, known_DDI_file, output_file, information_file, threshold):
    drug_information = read_drug_information(drug_information_file)    
    left_ddi_info, right_ddi_info = read_known_DDI_information(known_DDI_file)    
    similarity_df, similarity_info = read_similarity_file(similarity_file)
    DDI_prediction_df = pd.read_table(DDI_output_file)
    sentence_interaction_info, interaction_ddi_type_info = read_information_file(information_file)
    
    fp = open(output_file, 'w')
    print >>fp, '%s\t%s\t%s\t%s\t%s\t%s'%('Drug pair', 'Interaction type', 'Sentence', 'Score', 'Similar approved drugs (left)', 'Similar approved drugs (right)')
    cnt = 0
    s = time.time()
    for each_index, each_df in DDI_prediction_df.iterrows():
        cnt+=1
        if cnt % 10000 == 0:
            print cnt
            e = time.time()
            print 'Elapsed time: ', e-s
            
        drug_pair = each_df['Drug pair']
        left_drug, right_drug = drug_pair.split('_')
        new_DDI_type = str(each_df['DDI type'])
        DDI_type = interaction_ddi_type_info[new_DDI_type]
        sentence = each_df['Sentence']
        score = each_df['Score']
        
        left_corresponding_drugs = left_ddi_info[DDI_type]
        right_corresponding_drugs = right_ddi_info[DDI_type]

        left_drug_similarity_df = similarity_df.ix[left_drug][left_corresponding_drugs]
        left_selected_drugs = list(left_drug_similarity_df[left_drug_similarity_df>=threshold].index)
        
        right_drug_similarity_df = similarity_df.ix[right_drug][right_corresponding_drugs]
        right_selected_drugs = list(right_drug_similarity_df[right_drug_similarity_df>=threshold].index)

        left_drug_annotation_list = []
        for each_drug in left_selected_drugs:
            if each_drug in drug_information:
                targets = drug_information[each_drug]
                drug_target_information = '%s(%s)'%(each_drug, '|'.join(targets))
                left_drug_annotation_list.append(drug_target_information)

        right_drug_annotation_list = []
        for each_drug in right_selected_drugs:
            if each_drug in drug_information:
                targets = drug_information[each_drug]
                drug_target_information = '%s(%s)'%(each_drug, '|'.join(targets))
                right_drug_annotation_list.append(drug_target_information)

        left_drug_annotation_string = ';'.join(left_drug_annotation_list)
        right_drug_annotation_string = ';'.join(right_drug_annotation_list)        
        
        print >>fp, '%s\t%s\t%s\t%s\t%s\t%s'%(drug_pair, new_DDI_type, sentence, score, left_drug_annotation_string, right_drug_annotation_string)

    fp.close()        
    return


def summarize_prediction_outcome(result_file, output_file, information_file):    
    sentence_interaction_info, interaction_ddi_type_info = read_information_file(information_file)
    with open(result_file, 'r') as fp:
        fp.readline()
        out_fp = open(output_file, 'w')
        print >>out_fp, '%s\t%s\t%s\t%s'%('Drug pair', 'DDI type', 'Sentence', 'Score') 
        for line in fp:
            sptlist = line.strip().split('\t')
            drug_pair = sptlist[0].strip()
            drug_pair_list = drug_pair.split('_')
            drug1 = drug_pair_list[0]
            drug2 = drug_pair_list[1]
            DDI_class = sptlist[1].strip()
            predicted_score = sptlist[2].strip()
            new_interaction_type = sentence_interaction_info[DDI_class][2]
            if sentence_interaction_info[DDI_class][0] == '2':
                new_drug1 = drug2
                new_drug2 = drug1
            else:
                new_drug1 = drug1
                new_drug2 = drug2

            subject = sentence_interaction_info[DDI_class][0]
            template_sentence = sentence_interaction_info[DDI_class][1]
            prediction_outcome = template_sentence.replace('#Drug1', new_drug1)
            prediction_outcome = prediction_outcome.replace('#Drug2', new_drug2)
            print >>out_fp, '%s\t%s\t%s\t%s'%(drug_pair, new_interaction_type, prediction_outcome, predicted_score)                                
        out_fp.close()

    

