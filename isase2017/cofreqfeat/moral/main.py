#coding:utf-8

import random

if __name__ == '__main__':
    fout = open("alldatas", "w")
    with open("Positive") as f:
        for l in f:
            if l.startswith("+"):
                r = random.randint(1,4)
                if r == 4:
                    l = l.split("\n")[0]
                    l = l.split("+")[1]
                    fout.write(l + "\t" + "1\n")
    with open("Negative") as f:
        for l in f:
            if l.startswith("+"):
                l = l.split("\n")[0]
                l = l.split("+")[1]
                fout.write(l + "\t" + "0\n")
    fout.close()