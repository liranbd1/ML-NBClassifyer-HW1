from TrainModel import learn_NB_text
import Model as models
from Classifyer import ClassifyNB_text


def main():
    Pw, P = learn_NB_text()
    suc = ClassifyNB_text(Pw, P)
    print("the success rate is: {0}".format(suc))
    
if __name__ == "__main__":
    main()