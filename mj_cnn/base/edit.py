# coding:utf-8

c = 0
cs = 0
ss = 0
try:
    d_score = 0
    with open("nohup.out") as f:
        for l in f:
            c += 1
            l=l.split("\n")[0]
            s = float(l.split()[1])
            dev = float(l.split()[0])
            if ss < s:
                ss = s
            if dev > d_score:
                d_score = dev
                cs = c
                sss = s
except:
    a = 0

print cs
print d_score
print sss
print ss
