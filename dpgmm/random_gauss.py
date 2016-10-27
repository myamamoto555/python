# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
import scipy.special as ssp
import math
import copy
from scipy.stats import wishart

s1 = np.random.multivariate_normal([-3, 3], [[0.1, 0],[0, 0.1]], 200)
s2 = np.random.multivariate_normal([2, 4], [[0.1, 0],[0, 0.1]], 100)
s3 = np.random.multivariate_normal([0, 0], [[0.1, 0],[0, 0.1]], 100)
s4 = np.random.multivariate_normal([4, -2], [[0.1, 0],[0, 0.1]], 50)
s5 = np.random.multivariate_normal([-2, -4], [[0.1, 0],[0, 0.1]], 50)

inputs = np.r_[s1, s2, s3, s4, s5]

alpha = 1.0
beta = float(1)/3
df = 15
S = np.matrix([[0.1, 0], [0, 0.1]])
mu_zero = np.array([0.1,0.5])
# inputs = np.array([[1000,1000], [0,1], [10,9], [10,10], [255,255]])
d = 2

# クラスタw_iに所属するパターンの数をn_i
n = [len(inputs)]
# 全クラスタ数
c = 1
# 各観測パターンの所属クラスタを表す潜在変数の値
s = [0] * len(inputs)

P_max = -100000000

mu = [np.mean(inputs, axis=0)]
sigma = [np.cov(inputs, rowvar=0, bias=1)]

# probability density function
# p(x_k|theta_i)
def mnd(_x, _mu, _sig):
    x = np.matrix(_x)
    mu = np.matrix(_mu)
    sig = np.matrix(_sig)
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(x-mu)*sig.I*(x-mu).T)
    return np.exp(b)/a


def wishart_pdf(_x):
    w = wishart(df, S)
    density = w.pdf(_x[:,:,np.newaxis])
    return density

def get_eta(df, scale):
    w = ss.wishart(df, scale=scale)
    eta = w.rvs(1)
    return eta


def get_mu(mu_zero, eta):
    sample_mu = np.random.multivariate_normal(mu_zero,eta,1)
    return sample_mu


def get_inverse_matrix(A):
    inv_A = np.linalg.inv(A)
    return inv_A

for tt in range(1000):
    s_tmp = copy.deepcopy(s)
    c_tmp = copy.deepcopy(c)
    n_tmp = copy.deepcopy(n)
    mu_tmp = copy.deepcopy(mu)
    sigma_tmp = copy.deepcopy(sigma)
    #print s_tmp
    
    for k in range(len(inputs)):
        class_of_k = s[k]
        n[class_of_k] -= 1

        if n[class_of_k] == 0:
            c -= 1
            ttt = []
            for s_j in s:
                if s_j > class_of_k:
                    s_j -= 1
                ttt.append(s_j)
            del n[class_of_k]
            del mu[class_of_k]
            del sigma[class_of_k]
            s = ttt

        # 既存クラスタへの割り当て確率
        P = []
        for i in range(c):
            P.append(float(n[i]/(len(inputs)-1+alpha))*
                     mnd(inputs[k], mu[i], sigma[i]))

        # 新規クラスタへの割り当て確率
        vec = inputs[k] - mu_zero
        S_b_inv = get_inverse_matrix(S) + float(beta/(1+beta)) * (np.c_[vec] * vec)
        S_b = get_inverse_matrix(S_b_inv)
  
        right = (np.linalg.det(S_b)**float((df+1)/2) * ssp.gamma((df+1)/2)) / (np.linalg.det(S)**float(df/2) * ssp.gamma((df+1-d)/2))

        left = (beta / ((1 + beta) * math.pi)) ** float(d/2)

        P.append(float(alpha/(len(inputs)-1+alpha)) * left * right)
        P = np.array(P)
        #print P

        # 確率分布に従ってサンプリング
        class_sample =  np.random.multinomial(1, P / P.sum()).argmax()

        s[k] = class_sample
        #print "cs",class_sample
        #print "c", c
        # 既存クラスに割り当てられた場合
        if c > class_sample:
            n[class_sample] +=1
        # 新しいクラスに割り当てられた場合
        else:
            n.append(1)
            c += 1
            # 新しいパラメータの生成
            new_eta = get_eta(df, S)
            new_mu = get_mu(mu_zero, get_inverse_matrix(beta*new_eta))
            sigma.append(new_eta)
            mu.append(new_mu)
        #print s
        #print n


    # 各クラスタのパラメータの更新
    for i in range(c):
        x_sum = np.array([0.0, 0.0])
        x_num = 0
        for (j, inp) in enumerate(inputs):
            if s[j] == i:
                x_sum += inp 
                x_num += 1
        x_average = x_sum / float(x_num)
        #print x_average

        tmp = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        for (j, inp) in enumerate(inputs):
            if s[j] == i:
                xtmp = inp - x_average
                #print np.c_[xtmp]
                #print xtmp
                tmp += np.c_[xtmp] * xtmp
    
        nyu_c = df + n[i]
        S_q_inv = get_inverse_matrix(S) + tmp + (float(n[i] * beta) / (n[i] + beta)) * np.c_[x_average - mu_zero] * (x_average - mu_zero) 
        S_q = get_inverse_matrix(S_q_inv)

        mu_c = (n[i] * x_average + beta * mu_zero) / (n[i] + beta)
        ramda_c = (n[i] + beta) * sigma[i]

        mu[i] = get_mu(mu_c, get_inverse_matrix(ramda_c))
        sigma[i] = get_eta(nyu_c, S_q)

    """事後確率の計算"""
    # P(s)の計算
    # AFは不変なので、イーウェンスの抽出公式の分子のみ計算すれば良い
    # と思ったけど分子が大きくなりすぎるため、AFも計算
    AF = 0
    for i in range(len(inputs)):
        AF += np.log(i + 1)
    buntmp = 0
    for i in range(c):
        for j in range(1, n[i]):
            buntmp += np.log(j)
    logP_s = buntmp - AF

    # p(theta)p(x|theta,s)の計算
    tttmp = 0
    for i in range(c):
        G_0 = np.log(mnd(mu[i], mu_zero, S) * wishart_pdf(sigma[i]))
        liklihoodtmp = 0
        for j in range(len(inputs)):
            if s[j] == i:
                liklihoodtmp += np.log(mnd(inputs[j], mu[i], sigma[i]))
        tttmp += G_0 + liklihoodtmp
    print logP_s
    print tttmp
    posterior = logP_s + tttmp

    print posterior

    """事後確率最大化"""
    #print s_tmp
    if P_max < posterior:
        P_max = posterior
    else:
        s = copy.deepcopy(s_tmp)
        c = copy.deepcopy(c_tmp)
        n = copy.deepcopy(n_tmp)
        mu = copy.deepcopy(mu_tmp)
        sigma = copy.deepcopy(sigma_tmp)

    print "epoch",tt
    print s
    #print n
    print mu
