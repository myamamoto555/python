# -*- coding: utf-8 -*-

import pprint
import json
import corenlp



# パーサの生成
corenlp_dir = "/home/yamamoto/Downloads/stanford-corenlp-full-2013-06-20/"
propaty ="./user.properties"
parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir,properties=propaty)

# pos tagging and return orig
res = json.loads(parser.parse("cooking cookings dialogue dialogs"))
#words=res[u"sentences"][1][u"words"]
sents = res[u"sentences"]
for s in sents:
    words = s[u"words"]
    for w in words:
        print w[1][u"Lemma"]
pprint.pprint(res)



from nltk import stem
lemmatizer = stem.WordNetLemmatizer()

print lemmatizer.lemmatize('I am masahiro Yamamoto')
