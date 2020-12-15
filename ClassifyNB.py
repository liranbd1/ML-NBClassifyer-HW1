import Model as models
import numpy as np
from readTrainData import readTrainData

def classify_text(text, Pw, P):
    max_prob = np.NINF
    max_prob_label = ""
    for col_index in range(len(P)):
        prob_of_v = np.log(P[col_index])
        for word_index in range(len(text)): #row index
            if text[word_index] in models.VOC:
                row_index = np.where(models.VOC == text[word_index])
                prob_of_v = prob_of_v + np.log(Pw[row_index, col_index])
        if max_prob < prob_of_v:
            max_prob = prob_of_v
            max_prob_label = models.LABELS[col_index]
    return max_prob_label



def ClassifyNB_text(Pw, P):
    texAll, lbAll, voc, cat = readTrainData('r8-test-stemmed.txt')
    voc_list = list(voc)
    cat_list = list(cat)
    voc = np.array(voc_list)
    cat = np.array(cat_list)
    success_counter = 0
    for text_index in range(len(texAll)):
        label = classify_text(texAll[text_index], Pw, P)
        if label == lbAll[text_index]:
            success_counter = success_counter + 1
    
    #for each word in the text multiply P(w_i|y_j)
    # then multiply by the prior of v_j
    suc = 100*(success_counter/len(texAll))

    return suc