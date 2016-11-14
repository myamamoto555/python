#coding:utf-8
import math

def xor(x1, x2):
    if x1 != x2:
        return 1
    else:
        return 0

class evaluation(object):
    def __init__(self, predict_file, answer_file):
        self.predict = []
        self.predict_boundary = []
        self.answer = []
        self.answer_boundary = []
        self.datastore(predict_file, 'predict')
        self.datastore(answer_file, 'answer')
        self.dialogue_num = len(self.predict)

    def datastore(self, filename, listname):
        if listname == 'predict':
            lit = self.predict
            lit_b = self.predict_boundary
        else:
            lit = self.answer
            lit_b = self.answer_boundary

        with open(filename) as f:
            list_tmp = []
            list_b_tmp = []
            segment_tmp = -1
            for l in f:
                flag = l.split('\n')[0]
                if flag == '1':
                    segment_tmp += 1
                    list_tmp.append(segment_tmp)
                    list_b_tmp.append(int(flag))
                if flag == '0':
                    list_tmp.append(segment_tmp)
                    list_b_tmp.append(int(flag))
                if flag == '-1':
                    lit.append(list_tmp)
                    list_tmp = []
                    segment_tmp = -1
                    lit_b.append(list_b_tmp)
                    list_b_tmp = []
    
    # sentenceの数とwindow sizeが同じ場合、エラーが出る
    def calc_pk(self, window_size):
        self.p_k = 0
        k = window_size
        for p_list, a_list in zip(self.predict, self.answer):
            sum_tmp = 0
            N = len(p_list)
            for i in range(0, N - k):
                if p_list[i] == p_list[i+k]:
                    delta_p = 1
                else:
                    delta_p = 0
                if a_list[i] == a_list[i+k]:
                    delta_a = 1
                else:
                    delta_a = 0
                sum_tmp += xor(delta_p, delta_a)
            self.p_k += float(sum_tmp) / (N - k)
        self.p_k /= self.dialogue_num

    def calc_WD(self, window_size):
        self.WD = 0
        k = window_size
        for p_list, a_list in zip(self.predict_boundary, self.answer_boundary):
            sum_tmp = 0
            N = len(p_list)
            for i in range(0, N - k):
                b_p = sum(p_list[i:i+k])
                b_a = sum(a_list[i:i+k])
                sum_tmp += math.fabs(b_p - b_a)
            self.WD += float(sum_tmp) / (N - k)
        self.WD /= self.dialogue_num

if __name__ == '__main__':
    ev = evaluation('example.predict', 'example.answer')
    print ev.predict_boundary
    print ev.answer_boundary
    ev.calc_pk(1)
    ev.calc_WD(1)
    print ev.WD
