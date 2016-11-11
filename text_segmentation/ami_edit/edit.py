# coding:utf-8
import os
import json

# make filelist
filelist = []
rolelist = {}
with open('meeting.xml') as f:
    for l in f:
        if 'observation' in l:
            meeting_name = l.split('observation="')[1].split('"')[0] + '.'
        if 'role' in l:
            if meeting_name not in filelist:
                filelist.append(meeting_name)
            symbol = l.split('nxt_agent="')[1].split('"')[0]
            role = l.split('role="')[1].split('"')[0]
            if meeting_name not in rolelist:
                rolelist[meeting_name] = {}
                rolelist[meeting_name][symbol] = role
            else:
                rolelist[meeting_name][symbol] = role

person = ["A", "B", "C", "D"]
output_dir = 'plain/'


def decode(filename, p, start_word, end_word):
    sentence = ''
    flag = 0
    f = open('words/' + filename + p + '.words.xml')
    for l in f:
        if start_word in l or (flag == 1 and end_word not in l):
            w = l.split('>')[1].split('<')[0]
            if w!= '\n':
                sentence += w + ' '
            else:
                sentence = sentence[:-1]
            flag = 1
            if end_word == 'none':
                break
        if end_word in l:
            w = l.split('>')[1].split('<')[0]
            if w != '\n':
                sentence += w
            else:
                sentence = sentence[:-1]
            flag = 0
            break
    sentence = sentence.replace("&#39;", "'")
    return sentence


def get_topic_segment(filename):
    topic_segment = []
    flag = 0
    try:
        with open('topics/' + filename + 'topic.xml') as f:
            for l in f:
                if '#id' in l:
                    tmp_segment = l.split('#id(')[1].split(')')[0]
                if flag == 1:
                    topic_segment.append(tmp_segment)
                    flag = 0
                if 'nite:pointer' in l:
                    flag = 1
    except:
        topic_segment = ['*']
    return topic_segment


fout = open('ami_plain.jsonl', 'w')

for filename in filelist:
    topic_segment = get_topic_segment(filename)
    timedic = {}
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
                    if sentence != '':
                        if topic_segment == ['*']:
                            timedic[float(start_time)] = [rolelist[filename][p], sentence, '*']
                        else:
                            if start_word in topic_segment:
                                timedic[float(start_time)] = [rolelist[filename][p], sentence, '1']
                            else:
                                timedic[float(start_time)] = [rolelist[filename][p], sentence, '0']

    for k, v in sorted(timedic.items()):
        infodic = {}
        infodic['id'] = filename
        infodic['speaker'] = v[0]
        infodic['text'] = v[1]
        infodic['segment'] = v[2]
        
        jl = json.dumps(infodic, ensure_ascii=False)
        fout.write(jl + '\n')
fout.close()
