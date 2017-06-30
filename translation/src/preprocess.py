# coding: utf-8

def train_create(src, targ):
    src_vocs, src_indexes = convert_index(src)
    targ_vocs, targ_indexes = convert_index(targ)

    return src_vocs, src_indexes, targ_vocs, targ_indexes


def convert_index(fname):
    vocdic = {}
    with open(fname) as f:
        for i, l in enumerate(f):
            l = l.split("\n")[0]
            ws = l.split()
            for w in ws:
                if w in vocdic:
                    vocdic[w] += 1
                else:
                    vocdic[w] = 1
            if i > 1000000:
                break
    
    voc_list = []
    voc_list.append("<bos>")
    voc_list.append("<eos>")
    voc_list.append("<unk>")
    for key, value in vocdic.iteritems():
        if value > 2:
            voc_list.append(key)

    sent_indexes = []
    with open(fname) as f:
        for i, l in enumerate(f):
            sent_indexes_tmp = []
            sent_indexes_tmp.append(voc_list.index("<bos>"))
            l = l.split("\n")[0]
            ws = l.split()
            for w in ws:
                if w in voc_list:
                    sent_indexes_tmp.append(voc_list.index(w))
                else:
                    sent_indexes_tmp.append(voc_list.index("<unk>"))
            sent_indexes_tmp.append(voc_list.index("<eos>"))
            sent_indexes.append(sent_indexes_tmp)
            if i > 1000000:
                break
                    
    return voc_list, sent_indexes
