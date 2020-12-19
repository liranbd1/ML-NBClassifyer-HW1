# Liran Ben David - 308025444
# Roei Shenfeld - 206857955

from TrainModel import learn_NB_text
from Classifyer import ClassifyNB_text

def main():
    Pw, P = learn_NB_text()
    suc = ClassifyNB_text(Pw, P)
    print("the success rate is: {0}".format(suc))
    
if __name__ == "__main__":
    main()