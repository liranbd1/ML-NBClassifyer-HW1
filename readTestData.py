def readTestData(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    texAll = []
    lbAll = []
    for line in lines:
        splitted = line.split('\t')
        lbAll.append(splitted[0])
        texAll.append(splitted[1].split())
    
    return texAll, lbAll