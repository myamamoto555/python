#coding:utf-8

count = 0
fout1 = open("train1", "w")
fout2 = open("train2", "w")
fout3 = open("train3", "w")

with open("trainingset") as f:
    for l in f:
        if count < 100000:
            fout1.write(l)
        elif count < 200000:
            fout2.write(l)
        else:
            fout3.write(l)
        count += 1


fout1.close()
fout2.close()
fout3.close()
