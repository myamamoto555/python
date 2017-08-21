#coding:utf-8
allnum = 8
maxdevscore = 0
testscore = 0

for i in range(100, 5000, 100):
    fname = "./eval/test.hyp."
    dname = "./eval/dev.hyp."
    num = allnum - len(str(i))
    for j in range(num):
        fname += "0"
        dname += "0"
    fname += str(i)
    fname += ".bleu"
    dname += str(i)
    dname += ".bleu"
    with open(fname) as f:
        for l in f:
            tmptestscore = float(l.split("\n")[0])
            print i, str(tmptestscore),
    with open(dname) as f:
        for l in f:
            tmpdevscore = float(l.split("\n")[0])
            print str(tmpdevscore)
    if maxdevscore < tmpdevscore:
        maxdevscore = tmpdevscore
        testscore = tmptestscore
        ind = i

print ind, "maxdev: ", maxdevscore, "testscore: ", testscore
