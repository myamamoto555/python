# coding:utf-8
import codecs

filename = "ES2002a."
person = ["A", "B", "C", "D"]

timedic = {}

def decode(filename, p, start_word, end_word):
    sentence = ''
    flag = 0
    f = codecs.open('words/' + filename + p + '.words.xml')
    for l in f:
        if start_word in l or (flag == 1 and end_word not in l):
            w = l.split('>')[1].split('<')[0]
            if w!= '\n':
                sentence += w + ' '
            flag = 1
            if end_word == 'none':
                break
        if end_word in l:
            w = l.split('>')[1].split('<')[0]
            if w != '\n':
                sentence += w
            flag = 0
            break
    sentence = sentence.replace("&#39;", "'")
    return sentence


for p in person:
    with open('segments/' + filename + p + '.segments.xml') as f:
        for l in f:
            if "transcriber_start" in l:
                start_time = l.split('transcriber_start="')[1].split('"')[0]
                end_time = l.split('transcriber_end="')[1].split('"')[0]
            if "nite:child" in l:
                start_word = l.split('#id(')[1].split(')')[0]
                try:
                    end_word = l.split('..id(')[1].split(')')[0]
                except:
                    end_word = 'none'
                sentence = decode(filename, p, start_word, end_word)
                timedic[float(start_time)] = [p, sentence]

fout = open('out.txt', 'w')
for k, v in sorted(timedic.items()):
    print k,v
    #fout.write(v[0]+"\t"+v[1]+"\n")
    
fout.close()
