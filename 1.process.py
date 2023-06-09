import os
import numpy as np
def mergedata(rpath,wpath):
    fw = open(wpath,'w')
    files = os.listdir(rpath)
    for fn in files:
        fr = open(rpath+"/"+fn,'r')
        lines = fr.readlines()
        for line in lines:
            line = line.strip("\n")
            line = line.split("\t")
            data = ",".join(line)
            data = data + "\n"
            fw.write(data)
    fw.close()

def batchnormal(rpath,wpath):
    fw = open(wpath,'a')
    fr = open(rpath,'r')

    lines = fr.readlines()
    x,y,z,flag = [],[],[],[]
    for line in lines:
        line = line.split(',')
        #print(line[-4],line[-3],line[-2])
        x.append(float(line[-4]))
        y.append(float(line[-3]))
        z.append(float(line[-2]))
        flag.append(str(line[-1]))

    mean_little = [np.mean(x),np.mean(y),np.mean(z)]

    # the mean joint will be used in 3.accuracy.py and 4.predict.py
    print("--------mean--------")
    print(mean_little)

    for i,line in enumerate(lines):
        line = line.split(',')
        line = np.array(line[0:18])
        line = line.astype(np.float)
        line = line.reshape((-1,3))
        gap = line[5] - mean_little
        new_line = ''
        for vec in line:
            vec = vec - gap
            vec = vec.astype(np.str)
            vec_str = ",".join(vec)
            new_line = new_line+vec_str+','
        new_line = new_line + flag[i]
        fw.write(new_line)
    fw.close()




if __name__ == '__main__':
    rpath1 = "./rawdata/gestures1"
    wpath1 = "./rawdata/gestures1.txt"
    mergedata(rpath1,wpath1)

    rpath2 = "./rawdata/gestures2"
    wpath2 = "./rawdata/gestures2.txt"
    mergedata(rpath2, wpath2)

    rpath3 = "./rawdata/gestures3"
    wpath3 = "./rawdata/gestures3.txt"
    mergedata(rpath3, wpath3)

    batchnormal("./rawdata/gestures1.txt","./rawdata/gestures1_normal.txt")