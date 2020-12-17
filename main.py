from TrainModel import learn_NB_text
import Model as models
import ClassifyNB as nb
# Global variables:


# texAll = All of the texts each text is in a list
# lbAll = All of the labels 
# lbAll(i) is the label of texAll(i)
# voc = all the words no duplication
# cat = all the labels no duplication
# v_j is a label of the possible categories

def main():
    Pw, P = learn_NB_text()
    suc = nb.ClassifyNB_text(Pw, P)
    print("the success rate is: {0}".format(suc))

    
#    generate_words_dic(voc)
    #for key in  text_by_category.keys
     #   words_count[key]] = Counter(text)



if __name__ == "__main__":
    main()
    
    