from readTrainData import * 
from collections import Counter
import Model as model

# iteration over words or labels is done by looping over the vocabolary set and the category set respectavily (shorter loop - better performances)

# Calculating the priors of each label
def calc_priors(labels, train_data):
    priors = {}
    train_size = len(train_data) # The total size of the train data
    for label in labels: 
        occurences = train_data.count(label) # number of times we are seeing this label in the train data
        priors[label] = occurences/train_size # P(v_j)
    return priors

# Return a nested dictionary where top layer is by label key and lower layer is key by word, the final value is the 
# number of times word_k appeared in text_j (which is baiscally only the label)
def count_text_words(all_labels, train_labels, train_text):
    words_counter = {}
    for label in all_labels:
        texts = []
        index_list = [index for index, value in enumerate(train_labels) if value == label]
        for index in index_list:
            texts.extend(train_text[index])
        words_counter[label] = Counter(texts)
    return words_counter

# Return a dictionary where key is the tuple label, word and value is P(w_k|v_j)
def calc_p_w(all_words, words_counter, all_labels): 
    p_w = {}
    for label in all_labels:
        for word in all_words:
            n_k = words_counter[label][word]
            n = sum(words_counter[label].values())
            p_w[(label ,word)] = (n_k+1) / (n + len(all_words))
    return p_w

# Starting the learn process.
def learn_NB_text():
    train_text, train_label, all_words, all_labels = readTrainData("r8-train-stemmed.txt")
    word_counter = count_text_words(all_labels, train_label, train_text)
    model.VOC = all_words # Saving global variables in a different file to used and comapre in the test data (Not passing over data we didn't train for)
    model.LABELS = all_labels
    return calc_p_w(all_words, word_counter, all_labels), calc_priors(all_labels, train_label)