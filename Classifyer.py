import readTestData as rtd
import numpy as np
import Model as md

# Starting classifying the test text
# The calculation is using the lan function which is given from numpy.log, so insted of multiplication we can just sum values
def ClassifyNB_text(Pw, P):
    test_text , test_labels = rtd.readTestData("r8-test-stemmed.txt")
    success_counter = 0
    for (text, r_label) in zip(test_text, test_labels):
        max_prob_value = np.NINF # Set values 
        max_prob_label = ""
        for label in md.LABELS:
            prob_value = 0
            prob_value = prob_value + np.log(P[label]) # ln(P(v_j))
            for word in text:
                if word in md.VOC: # Making sure the word is in the volcabolary
                    prob_value = prob_value + np.log(Pw[(label, word)]) #ln(P(w_k|v_j))
            if prob_value > max_prob_value: # Comparing probabilities 
                max_prob_value = prob_value # Saving highest value for next comparsion
                max_prob_label = label # Saving highest value label
        if r_label == max_prob_label: # Checking if we got the right label
            success_counter = success_counter + 1 # If yes increase success counter
    
    return 100*success_counter / len(test_labels)