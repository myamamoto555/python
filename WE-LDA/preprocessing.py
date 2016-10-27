# coding:utf-8
import os
import word2vec
import json
import corenlp

stopwords_list = "a,s,able,about,above,according,accordingly,across,actually,after,afterwards,again,against,ain,t,all,allow,allows,almost,alone,along,already,also,although,always,am,among,amongst,an,and,another,any,anybody,anyhow,anyone,anything,anyway,anyways,anywhere,apart,appear,appreciate,appropriate,are,aren,t,around,as,aside,ask,asking,associated,at,available,away,awfully,be,became,because,become,becomes,becoming,been,before,beforehand,behind,being,believe,below,beside,besides,best,better,between,beyond,both,brief,but,by,c,mon,c,s,came,can,can,t,cannot,cant,cause,causes,certain,certainly,changes,clearly,co,com,come,comes,concerning,consequently,consider,considering,contain,containing,contains,corresponding,could,couldn,t,course,currently,definitely,described,despite,did,didn,t,different,do,does,doesn,t,doing,don,t,done,down,downwards,during,each,edu,eg,eight,either,else,elsewhere,enough,entirely,especially,et,etc,even,ever,every,everybody,everyone,everything,everywhere,ex,exactly,example,except,far,few,fifth,first,five,followed,following,follows,for,former,formerly,forth,four,from,further,furthermore,get,gets,getting,given,gives,go,goes,going,gone,got,gotten,greetings,had,hadn,t,happens,hardly,has,hasn,t,have,haven,t,having,he,he,s,hello,help,hence,her,here,here,s,hereafter,hereby,herein,hereupon,hers,herself,hi,him,himself,his,hither,hopefully,how,howbeit,however,i,d,i,ll,i,m,i,ve,ie,if,ignored,immediate,in,inasmuch,inc,indeed,indicate,indicated,indicates,inner,insofar,instead,into,inward,is,isn,t,it,it,d,it,ll,it,s,its,itself,just,keep,keeps,kept,know,knows,known,last,lately,later,latter,latterly,least,less,lest,let,let,s,like,liked,likely,little,look,looking,looks,ltd,mainly,many,may,maybe,me,mean,meanwhile,merely,might,more,moreover,most,mostly,much,must,my,myself,name,namely,nd,near,nearly,necessary,need,needs,neither,never,nevertheless,new,next,nine,no,nobody,non,none,noone,nor,normally,not,nothing,novel,now,nowhere,obviously,of,off,often,oh,ok,okay,old,on,once,one,ones,only,onto,or,other,others,otherwise,ought,our,ours,ourselves,out,outside,over,overall,own,particular,particularly,per,perhaps,placed,please,plus,possible,presumably,probably,provides,que,quite,qv,rather,rd,re,really,reasonably,regarding,regardless,regards,relatively,respectively,right,said,same,saw,say,saying,says,second,secondly,see,seeing,seem,seemed,seeming,seems,seen,self,selves,sensible,sent,serious,seriously,seven,several,shall,she,should,shouldn,t,since,six,so,some,somebody,somehow,someone,something,sometime,sometimes,somewhat,somewhere,soon,sorry,specified,specify,specifying,still,sub,such,sup,sure,t,s,take,taken,tell,tends,th,than,thank,thanks,thanx,that,that,s,thats,the,their,theirs,them,themselves,then,thence,there,there,s,thereafter,thereby,therefore,therein,theres,thereupon,these,they,they,d,they,ll,they,re,they,ve,think,third,this,thorough,thoroughly,those,though,three,through,throughout,thru,thus,to,together,too,took,toward,towards,tried,tries,truly,try,trying,twice,two,un,under,unfortunately,unless,unlikely,until,unto,up,upon,us,use,used,useful,uses,using,usually,value,various,very,via,viz,vs,want,wants,was,wasn,t,way,we,we,d,we,ll,we,re,we,ve,welcome,well,went,were,weren,t,what,what,s,whatever,when,whence,whenever,where,where,s,whereafter,whereas,whereby,wherein,whereupon,wherever,whether,which,while,whither,who,who,s,whoever,whole,whom,whose,why,will,willing,wish,with,within,without,won,t,wonder,would,would,wouldn,t,yes,yet,you,you,d,you,ll,you,re,you,ve,your,yours,yourself,yourselves,zero".split(',')
stopwords_list = ""

class DATA:
    def __init__(self, documents_directory_path):
	print "vector model creation"
	vecmodel = word2vec.vecmodelcreate()
	print "vector model creation done"

	# generation of parser
	corenlp_dir = "/home/yamamoto/Downloads/stanford-corenlp-full-2013-06-20/"
	propaty ="./user.properties"
	parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir,properties=propaty)

        fis = os.listdir(documents_directory_path)
        self.docs = []
        self.vocs = []
	for fi in fis:
            with open(documents_directory_path+fi) as f:
		doc_ws = []
		for l in f:
		    try:
		        res = json.loads(parser.parse(l))
		    except:
			res = ""
		    if res != "":
		        sents = res[u"sentences"]
		        for s in sents:
    		            words = s[u"words"]
    		            for w in words:
        		        w_lemma = w[1][u"Lemma"]
                                if not w_lemma in stopwords_list and not w_lemma in self.vocs:
			            if w_lemma in vecmodel:
                                        self.vocs.append(w_lemma)
			        if w_lemma in self.vocs:
			            doc_ws.append(self.vocs.index(w_lemma))
		self.docs.append(doc_ws)
	self.filesave("testdocs/")

    def filesave(self, dp):
	foutdoc = open(dp+"data.doc", "w")
        foutvoc = open(dp+"data.voc", "w")

        for v in self.vocs:
	    foutvoc.write(v+"\n")
	for d in self.docs:
	    for w in d:
		foutdoc.write(str(w)+" ")
	    foutdoc.write("\n")
	foutvoc.close()

if __name__ == '__main__':
	data = DATA("./docs/")
	print len(data.vocs)	

