# coding:utf-8
import numpy as np
import preprocessing
import math

np.random.seed(0)

class LDA:
    def __init__(self, alpha, beta, gamma, K, data):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.D = len(data.docs)
        self.K = K
        self.V = len(data.vocs)
        self.A = data.authors_count
        self.docs = data.docs
        self.authors = data.authors

        self.m_y = np.zeros(self.K) + self.V * self.beta  # word count of topic y
        self.n_csty = []
        self.m_yv = np.zeros((self.K, self.V)) + self.beta  # word count of topic y and vocabulary v

        self.z_dsn = []  # topic of sentence s in document d and word n
	self.l_ct = []
        self.c_st = []
        self.s_a = np.zeros((self.A, 2))  # count of speaker a and segment "x" (x = 0 or 1)

        self.init_segment_assign()
	self.init_topic_assign()

        self.params_update()

    def init_segment_assign(self):
        for i, d in enumerate(self.docs):
            l_c = []
            for j, s in enumerate(d):
                author = self.authors[i][j]
                if j == 0:
                    l_c.append(1)
                    self.s_a[author][1] += 1
                else:
                    l_c.append(1)
                    self.s_a[author][1] += 1
            self.l_ct.append(l_c)

    def params_update(self):
        self.n_csty = []
        self.c_st = []
        for i, (l_c, z_sn) in enumerate(zip(self.l_ct, self.z_dsn)):
            indexes = [i for i, x in enumerate(l_c) if x == 1]
            n_sty = []
            st = []
            for j in range(len(indexes)):
                tmp = []
                n_y = []
                if j+1 == len(indexes):
                    for k in range(indexes[j], len(l_c)):
                        tmp.extend(z_sn[k])
                        st.append(j)
                else:
                    for k in range(indexes[j], indexes[j+1]):
                        tmp.extend(z_sn[k])
                        st.append(j)
                for k in range(self.K):
                    n_y.append(tmp.count(k) + self.alpha)
                n_sty.append(n_y)
            self.n_csty.append(np.array(n_sty))
            self.c_st.append(st)

    def init_topic_assign(self):
        for i, d in enumerate(self.docs):
            z_sn = []
            for j, s in enumerate(d):
                z_n = []
                for n in s:
                    z = np.random.randint(0, self.K)
		
                    z_n.append(z)
                    self.m_y[z] += 1
                    self.m_yv[z, n] += 1
                z_sn.append(z_n)
	    self.z_dsn.append(np.array(z_sn))

    def learning_z(self):
        for i, d in enumerate(self.docs):
            z_sn = self.z_dsn[i]
            n_sty = self.n_csty[i]
            for j, s in enumerate(d):
                st = self.c_st[i][j]
                n_y = n_sty[st]
                z_n = z_sn[j]
                for l, n in enumerate(s):
                    z = z_n[l]
                    # discount
                    n_y[z] -= 1
                    self.m_yv[z, n] -= 1
                    self.m_y[z] -= 1
                
                    # sampling
                    p_z = (self.m_yv[:, n] / self.m_y) * (n_y / n_y.sum())
                    new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                
                    # update
                    z_n[l] = new_z
                    n_y[new_z] += 1
                    self.m_yv[new_z, n] += 1
                    self.m_y[new_z] += 1
    
    def learning_l(self):
        for i, d in enumerate(self.docs):
            for j, s in enumerate(d):
                if j > 0:
                    author = self.authors[i][j]
                    l = self.l_ct[i][j]
                    p_l = []
                    # discount
                    self.s_a[author][l] -= 1

                    # probability of l_ct = 0
                    self.l_ct[i][j] = 0
                    self.params_update()
                    n_sty = self.n_csty[i]
                    st = self.c_st[i][j]
                    n_y = n_sty[st]

                    p0 = ((self.s_a[author][0] + self.gamma) / \
                          (self.s_a.sum() + 2 * self.gamma)) * \
                        math.exp(reduce(lambda x,y:x+y, [math.lgamma(n_y[k]) for k in range(self.K)]) - math.lgamma(n_y.sum()))

                    # probability of l_ct = 1
                    self.l_ct[i][j]= 1
                    self.params_update()
                    n_sty = self.n_csty[i]
                    st = self.c_st[i][j]
                    n_y = n_sty[st]
                    n_y_bef = n_sty[st-1]

                    p1 = ((self.s_a[author][1] + self.gamma) / \
                          (self.s_a.sum() + 2 * self.gamma)) * \
                        (math.gamma(self.K * self.alpha) / \
                         math.gamma(self.K ** self.alpha)) * \
                        math.exp(reduce(lambda x,y:x+y, [math.lgamma(n_y[k]) for k in range(self.K)]) - math.lgamma(n_y.sum())) * \
                        math.exp(reduce(lambda x,y:x+y, [math.lgamma(n_y_bef[k]) for k in range(self.K)]) - math.lgamma(n_y_bef.sum()))


                    p_l.append(p0)
                    p_l.append(p1)
                    p_l = np.array(p_l)
                    new_l = np.random.multinomial(1, p_l / p_l.sum()).argmax()

                    # update
                    self.s_a[author][new_l] += 1
                    self.l_ct[i][j]= new_l
                    self.params_update()

    def topic_word_dist(self):
        return self.m_yv / self.m_y[:, np.newaxis]

    def perplexity(self):
        phi = self.topic_word_dist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for i, d in enumerate(self.docs):
            n_sty = self.n_csty[i]
            for j, s in enumerate(d):
                st = self.c_st[i][j]
                n_y = n_sty[st]
                theta = n_y / (len(self.docs[i]) + Kalpha)
                for l, n in enumerate(s):
                    log_per -= np.log(np.inner(phi[:, n], theta))
                N += len(s)
        return np.exp(log_per / N)
    

if __name__ == '__main__':
    K = 10
    print "preprocessing"
    filename = "debate2008"
    data = preprocessing.DATA("./datas/"+filename)
    print "preprocessing done"

    lda = LDA(0.5, 0.5, 0.5, K, data)
    for i in range(5000):
        lda.learning_z()
        lda.learning_l()
        print i, lda.perplexity()
        print lda.l_ct
