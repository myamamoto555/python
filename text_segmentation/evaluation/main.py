#coding:utf-8

class evaluation(object):
    def __init__(self, predict_file, answer_file):
        self.predict = []
        self.answer = []
        self.datastore(predict_file, 'predict')
        self.datastore(answer_file, 'answer')
    
    def datastore(self, filename, listname):
        if listname == 'predict':
            lit = self.predict
        else:
            lit = self.answer

        with open(filename) as f:
            list_tmp = []
            for l in f:
                flag = l.split('\n')[0]
                if flag == '1' or flag == '0':
                    list_tmp.append(flag)    
                if flag == '-1':
                    lit.append(list_tmp)
                    list_tmp = []


if __name__ == '__main__':
    ev = evaluation('example.predict', 'example.answer')
    print ev.predict
    print ev.answer
