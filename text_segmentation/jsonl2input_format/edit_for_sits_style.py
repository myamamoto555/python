# -*- coding: utf-8 -*-
import json

class convdatas(object):
    def __init__(self, filename):
        self.filename = filename
        self.voc = []
        self.speakers = []
        self.conv_num = 0
        self.sentence_num = 0

    def save_voc(self):
        with open(self.filename) as f:
            for l in f:
                data = json.loads(l)
                words = data['text']
                for w in words:
                    if w not in self.voc:
                        self.voc.append(w)

        fout = open('ldaformat/ami.voc', 'w')
        for v in self.voc:
            fout.write(v + '\n')
        fout.close()

    def save_whois(self):
        with open(self.filename) as f:
            for l in f:
                data = json.loads(l)
                speaker = data['speaker']
                if speaker not in self.speakers:
                    self.speakers.append(speaker)

        fout = open('ldaformat/ami.whois', 'w')
        for s in self.speakers:
            fout.write(s + '\n')
        fout.close()

    def save_shows(self):
        docids = []
        befid = 'tmpid'
        with open(self.filename) as f:
            for l in f:
                data = json.loads(l)
                docid = data['id']
                if befid == docid:
                    docids.append(docid)
                else:
                    docids.append('')
                    docids.append(docid)
                    self.conv_num += 1
                befid = docid
        docids = docids[1:]
        self.sentence_num = len(docids)

        fout = open('ldaformat/ami.shows', 'w')
        for d in docids:
            fout.write(d + '\n')
        fout.close()

    def save_authors(self):
        authors = []
        befid = 'tmpid'
        with open(self.filename) as f:
            for l in f:
                data = json.loads(l)
                docid = data['id']
                author = str(self.speakers.index(data['speaker']))
                if befid == docid:
                    authors.append(author)
                else:
                    authors.append('-1')
                    authors.append(author)
                befid = docid

        authors.append('-1')
        authors = authors[1:]
        fout = open('ldaformat/ami.authors', 'w')
        for a in authors:
            fout.write(a + '\n')
        fout.close()

    def save_words(self):
        fout = open('ldaformat/ami.words', 'w')
        fout.write(str(self.conv_num) + '\n')
        fout.write(str(self.sentence_num) + '\n')
        befid = ''
        with open(self.filename) as f:
            for l in f:
                data = json.loads(l)
                words = data['text']
                docid = data['id']
                if befid != docid and befid != '':
                    fout.write('\n')
                fout.write(str(len(words)) + '\t')
                for w in words:
                    fout.write(str(self.voc.index(w)) + ' ')
                fout.write('\n')
                befid = docid
        fout.write('\n')
        fout.close()

    def save_text(self):
        fout = open('ldaformat/ami.text', 'w')
        befid = ''
        with open(self.filename) as f:
            for l in f:
                data = json.loads(l)
                words = data['text']
                speaker = data['speaker']
                docid = data['id']
                if befid != docid and befid != '':
                    sentence_count += 1
                    fout.write(str(sentence_count)+'\t'+speaker+'\t'+str(words)+'\n')
                else:
                    fout.write('\n')
                    sentence_count = 0
                    fout.write(str(sentence_count)+'\t'+speaker+'\t'+str(words)+'\n')
                    befid = docid
        
        fout.write('\n')
        fout.close()
                    

if __name__ == '__main__':
    filename = 'ami_plain.jsonl'
    cd = convdatas(filename)
    cd.save_voc()
    cd.save_whois()
    cd.save_shows()
    cd.save_authors()
    cd.save_words()
    cd.save_text()
