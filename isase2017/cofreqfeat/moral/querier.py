#coding:utf-8
import urllib2

# 共起取得
def getfreq(query):
    qorig = "http://ssgnc.isoft/idc.cgi?qtype=COFREQ&word="
    response = urllib2.urlopen(qorig + query)
    c = 0
    for r in response:
        if c == 1:
            freq = r.split(",")[0]
        c += 1
        
    return freq


# 1単語の頻度を取得
def getfreq2(query):
    qorig = "http://ssgnc.isoft/idc.cgi?qtype=FREQ&word="
    response = urllib2.urlopen(qorig + query)
    c = 0
    for r in response:
        if c == 1:
            freq = r.split(",")[0]
        c += 1

    return freq


if __name__ == '__main__':
    query = "褒める,恋"
    print getfreq(query)
    print getfreq2("文")
