# modelsummary.txtを開いて、モデルのTotal paramsを検索し、その値を返す関数get_params()

import os
import numpy as np
def get_params():
    #os.remove('parameters.txt')
    with open('modelsummary.txt','r',encoding='utf-8') as filetxt:
        lines=filetxt.readlines()
        #print(lines)
        for line in lines:
            line = line.split('\n')[0]
            #print(line)
            if 'Total params:' in line:
                #print(line)
                parameter = line.split(': ')[1].replace(',','')
                parameter = np.array(parameter,dtype=np.float)
                #fb = open('parameters.txt','a')
                #fb.write('Parameters:%sM'%(parameter/1e6))
                #fb.write('\n')
                #print('%s'% (a/1e6))

    #fb.close()
    filetxt.close()
    return parameter
