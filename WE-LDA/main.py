# coding:utf-8
import numpy as np
import preprocessing
import doc

np.random.seed()

class LDA:
    def __init__(self, alpha, beta, sigma_zero, sigma, mu, K, C, data):
        self.alpha = alpha
        self.beta = beta
        self.sigma_zero = sigma_zero
        self.sigma = sigma
        self.mu = mu
        self.D = len(data.docs)
        self.K = K
        self.V = len(data.vocs)
        self.C = C
        self.docs = data.docs
        self.vectors = data.vectors

        self.n_k = np.zeros(self.K) + self.C * self.beta  # word count of topic k
        self.n_dk = np.zeros((self.D, self.K)) + self.alpha  # word count of document d and topic k
        self.n_kc = np.zeros((self.K, self.C)) + self.beta  # word count of topic k and vocabulary v
        self.n_c = np.zeros(self.C)

        self.z_dn = []  # topic of document d and word n
	self.c_dn = []  # concept of document d and word n

        self.init_concept_assign()
	self.init_topic_assign()

    def init_concept_assign(self):
        for i, d in enumerate(self.docs):
            c_n = []
            for n in d:
                c = np.random.randint(self.C)
		c_n.append(c)
		self.n_c[c] += 1
	    self.c_dn.append(np.array(c_n))

    def init_topic_assign(self):
        for i, d in enumerate(self.docs):
            z_n = []
            for j, n in enumerate(d):
                c = self.c_dn[i][j]
                p_z = self.n_kc[:, c] * self.n_dk[i] / self.n_k
		z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
		
		z_n.append(z)
		self.n_k[z] += 1
            	self.n_dk[i, z] += 1
		self.n_kc[z, c] += 1
	    self.z_dn.append(np.array(z_n))

    def learning(self):
        for i, d in enumerate(self.docs):
            z_n = self.z_dn[i]
            c_n = self.c_dn[i]
	    n_dk = self.n_dk[i]
            for j, n in enumerate(d):
		# discount
                c = c_n[j]
		z = z_n[j]
		n_dk[z] -= 1
		self.n_kc[z, c] -= 1
		self.n_k[z] -= 1
                self.n_c[c] -= 1
                
		# sampling
		p_z = self.n_kc[:, c] * n_dk / self.n_k
		new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                
		# update
                z_n[j] = new_z
		n_dk[new_z] += 1
		self.n_k[new_z] += 1


                # sampling
                I = np.matrix(np.identity(300))
                p_c = self.n_kc[new_z:] * [self.mnd(self.vectors[int(n)], self.get_mu_c[c], self.get_sigma_c[c] * I) for c in range(self.C)]
                new_c = np.random.multinomial(1, p_c / p_c.sum()).argmax()
                
                # update
                self.n_kc[new_z, new_c] += 1
                self.n_c[new_c] += 1
                c_n[j] = new_c
    
    def get_mu_c(self, c):
        mu_c = (self.sigma * self.mu + self.sigma * average) / (self.sigma + self.n_c[c] * sigma_zero)
        return mu_c

    def get_sigma_c(self, c):
        average = np.zeros(300)
        num = 0
        for i, d in enumerate(self.docs):
            for j, n in enumerate(d):
                if self.c_dn[i, j] == c:
                    average += self.docs.vectors[int(n)]
                    num += 1
        average /= float(num)
        sigma_c = (1 + (self.sigma_zero / (self.n_c[c] * sigma_zero) + sigma)) * self.sigma
                
        return sigma_c
    
    # probability density function
    # p(x_k|theta_i)
    def mnd(_x, _mu, _sig):
        x = np.matrix(_x)
        mu = np.matrix(_mu)
        sig = np.matrix(_sig)
        a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
        b = np.linalg.det(-0.5*(x-mu)*sig.I*(x-mu).T)
        return np.exp(b)/a
        
    def topic_word_dist(self):
	return self.n_kv / self.n_k[:, np.newaxis]
    
    def perplexity(self, docs=None):
	if docs == None:
	    docs = self.docs
	phi = self.topic_word_dist()
	log_per = 0
	N = 0
	Kalpha = self.K * self.alpha
	for i, doc in enumerate(docs):
	    theta = self.n_dk[i] / (len(self.docs[i]) + Kalpha)
	    for n in doc:
		log_per -= np.log(np.inner(phi[:, n], theta))
	    N += len(doc)
        return np.exp(log_per / N)

if __name__ == '__main__':
    K = 8
    alpha = 0.1
    beta = 0.01
    sigma_zero = 1.0
    sigma = 0.5
    C = 20

    print "preprocessing"
    data = doc.data("./testdocs/data")
    print "preprocessing done"

    mu = data.average_vector()

    lda = LDA(alpha, beta, sigma_zero, sigma, mu, K, C,  data)
    for i in range(10):
        lda.learning()
        #print lda.perplexity()
