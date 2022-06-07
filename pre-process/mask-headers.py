import pandas as pd
import os

rounds = [1,3,4]

for round in rounds:
    
    file_path = 'FEDORA-ICDM/raw_data/Semtab2019/data/Round '+str(round)+'/tables/'
    OUTPUT_PATH = 'FEDORA/data/no-headers/Semtab2019/Round'+str(round)+'/'

    headers = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    i = 0
    error  = []
    for infile in os.listdir(file_path):
        #print(i)
        i+=1
        path = file_path+infile
        out_path = OUTPUT_PATH+infile
        try:
            df=pd.read_csv(path, header=0)
        except pd.errors.ParserError:
            error.append(infile)
            print(infile)
            print(i)
            continue
        length = df.shape[1]
        try:
            df.columns=headers[:length]
        except ValueError:
            error.append(infile)
            print(infile)
            print(i)
            continue
        df.to_csv(out_path, index=False)


    print(error)
