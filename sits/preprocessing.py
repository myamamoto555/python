# coding:utf-8
import os

stopwords_list = []

class DATA:
    def __init__(self, filename):
        self.docs = []
        self.vocs = []
        self.authors = []
        self.authors_count = 0
        self.filename = filename

        self.docs_store()
        self.authors_store()
        self.vocs_store()
        self.authors_count_store()

    def docs_store(self):
        with open(self.filename+".words") as f:
            for i, l in enumerate(f):
                if i > 1:
                    if i == 2:
                        doctmp = []
                    if l == "\n":
                        self.docs.append(doctmp)
                        doctmp = []
                    else:
                        l = l.split("\n")[0]
                        ws = l.split("\t")[1].split()
                        doctmp.append(ws)

    def authors_store(self):
        with open(self.filename+".authors") as f:
            for i, l in enumerate(f):
                author = l.split("\n")[0]
                if i == 0:
                    authortmp = []
                if author == "-1":
                    self.authors.append(authortmp)
                    authortmp = []
                else:
                    authortmp.append(author)

    def vocs_store(self):
        with open(self.filename+".voc") as f:
            for l in f:
                w = l.split("\n")[0]
                self.vocs.append(w)
                    
    def authors_count_store(self):
        self.authors_count = sum(1 for line in open(self.filename+".whois"))


if __name__ == '__main__':
    filename = "debate2008"
    data = DATA("./datas/"+filename)
    print len(data.docs)
    print len(data.authors)
    print len(data.vocs)
    print data.authors_count
