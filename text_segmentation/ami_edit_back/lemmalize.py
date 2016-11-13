# -*- coding: utf-8 -*-
import pprint
import json
import corenlp

def get_lemma_list(sentence):
    # パーサの生成
    corenlp_dir = "/home/yamamoto/Downloads/stanford-corenlp-full-2013-06-20/"
    propaty ="./user.properties"
    parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir,properties=propaty)

    lemma_list = []
    sents = json.loads(parser.parse(sentence))[u"sentences"]
    for s in sents:
        words = s[u"words"]
        for w in words:
            lemma_list.append(w[1][u"Lemma"])
    return lemma_list


def main():
    print get_lemma_list("cooking cookings dialogue dialogs")


if __name__ == '__main__':
    main()


