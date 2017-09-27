#coding:utf-8

import MeCab

tagger = MeCab.Tagger("-Owakati")
patrain = open('train.sent', 'w')
patest = open('test.sent', 'w')
padev = open('dev.sent', 'w')
antrain = open('train.ans', 'w')
antest = open('test.ans', 'w')
andev = open('dev.ans', 'w')

nc = 0
pc = 0

with open('train.txt') as f:
    for l in f:
        l = l.split("\n")[0]
        sent = l.split("\t")[0]
        score = l.split("\t")[1]
        if float(score) == -1:
            score = str(0)
        res = tagger.parse(sent).split("\n")[0]
        if nc < 1000 and score == '0':
            padev.write(res+"\n")
            andev.write(score+"\n")
            nc += 1
        elif pc < 1000 and score == '1':
            padev.write(res+"\n")
            andev.write(score+"\n")
            pc += 1
        else:
            patrain.write(res+"\n")
            antrain.write(score+"\n")

patrain.close()
antrain.close()
padev.close()
andev.close()

with open('test.txt') as f:
    for l in f:
        l = l.split("\n")[0]
        sent = l.split("\t")[0]
        score = l.split("\t")[1]
        if float(score) > 3:
            score = str(1)
        else:
            score = str(0)
        res = tagger.parse(sent).split("\n")[0]
        patest.write(res+"\n")
        antest.write(score+"\n")

patest.close()
antest.close()


