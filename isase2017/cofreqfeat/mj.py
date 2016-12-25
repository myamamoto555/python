# -*- coding: utf-8 -*-
import querier

def dicload():
    nw = []
    pw = []
    fp = open("dic/pos")
    for l in fp:
        l = l.split("\n")[0]
        pw.append(l)

    fn = open("dic/neg")
    for l in fn:
        l = l.split("\n")[0]
        nw.append(l)
    return pw, nw


def getcofreq(query):
    return float(querier.getfreq(query))


def getscore(pw , nw , orig):
    orig_query = ""
    for o in orig:
        if orig_query != "":
            orig_query += "," + o
        else:
            orig_query += o

    p_score = []
    n_score = []
    for p in pw:
        query = p + "," + orig_query
        p_score.append(getcofreq(query))
    for n in nw:
        query = n + "," + orig_query
        n_score.append(getcofreq(query))
    return p_score, n_score


# 頻度の一番大きい単語を削除する
# 論文で文簡約って言っているところ
def remove_word(orig):
    max_freq = 0
    max_freq_word = ""
    for o in orig:
        freq = float(querier.getfreq2(o))
        if freq > max_freq:
            max_freq = freq
            max_freq_word = o
    # print "remove: " + max_freq_word
    orig.remove(max_freq_word)
    return orig


def moraljudgment(orig):
    # load dictionary
    pw, nw = dicload()
    flag = 0
    while True:
        # calculate score
        p_score, n_score = getscore(pw, nw, orig)
        for p, n in zip(p_score, n_score):
            if p != 0 or n != 0:
                flag = 1
        if flag == 0 and len(orig) > 1:
            orig = remove_word(orig)
        else:
            break
    score = []
    score.extend(p_score)
    score.extend(n_score)

    if flag == 0:
        for i in range(40):
            score.append(0)
    return score

    # final score calculation
    # 値は-1 ~ 1
            
    #final_score = (p_score - n_score) / (p_score + n_score)
    #print final_score

    # 実装していないところ               
    # 否定語の処理をいれたほうがいい
    # if 入力文中に"ない"がある場合、final_scoreを-1倍する。

        
if __name__ == "__main__":
    # 分かち書きして内容語の原型をあらかじめ取得
    orig = ["言辞", "並べる"]

    moraljudgment(orig)
