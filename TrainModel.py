from readTrainData import * 
import numpy as np
from collections import Counter
import Model as model

text_by_category = {}
words_count = {}
labels_prob = {}

def calculate_p_v_j(docs, examples):
    return len(docs) / len(examples)

def calculate_p_w_k_given_v_j( volcabolary, words_counter, labels):
    for k in range(len(volcabolary)):
        for j in range(len(labels)):
            n = len(text_by_category[labels[j]])
            n_k = words_counter[labels[j]][volcabolary[k]]
            model.WORD_LABEL_PROB[k,j] = (n_k + 1)/ (n+ len(volcabolary))
        
def concatinate_text(docs):
    concatined_text_list = list()
    for text_list in docs:
        concatined_text_list = concatined_text_list + text_list
    return concatined_text_list

def learn_NB_text():
    texAll, lbAll, voc, cat = readTrainData("r8-train-stemmed.txt")
    voc_list = list(voc)
    cat_list = list(cat)
    model.VOC = np.array(voc_list)
    model.LABELS = np.array(cat_list)
    model.WORD_LABEL_PROB = np.zeros((len(voc), len(cat)))
    for category in model.LABELS:
        docs = list()
        for index in range(len(lbAll)):
            if category == lbAll[index]:
                docs.append(texAll[index])
        labels_prob[category] = calculate_p_v_j(docs, texAll)
        text_by_category[category] = concatinate_text(docs) 
    
    for key in text_by_category.keys():
        words_count[key] = Counter(text_by_category[key])
    
    calculate_p_w_k_given_v_j(model.VOC, words_count, model.LABELS)
    model.LABEL_PROB = np.zeros(len(model.LABELS))
    for index in range(model.LABELS.size):
        model.LABEL_PROB[index] = labels_prob[model.LABELS[index]]
        
    return model.WORD_LABEL_PROB, model.LABEL_PROB