# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import numpy as np
import os
import copy

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/bin'
use_gpu = True


def _array_module():
    return chainer.cuda.cupy if use_gpu else numpy


def _mkivar(array):
    m = _array_module()
    return chainer.Variable(m.array(array, dtype=m.int32))


def _zeros(shape):
    m = _array_module()
    return chainer.Variable(m.zeros(shape, dtype=m.float32))


# return top-k index and scores. This is used for beam search.
# example                                                                
# input: x = np.array([[1, 2, 3], [4, 5, 6]]), k=2                                                   
# return: ([(1, 2), (1, 1)], [6, 5])                                                                 
def top_k(xs, k, b):
    ids_list = []
    scores_list = []
    for b in range(b):
        x = xs[k*b:k*(b+1)]
        for i in range(k):
            ids = np.unravel_index(x.argmax(), x.shape)
            score = x[ids[0]][ids[1]]
            ids_list.append((ids[0]+k*b, ids[1]))
            scores_list.append(score)
            x[ids[0]][ids[1]] = -1000000
    return ids_list, scores_list


def top_k_init(xs, k, b):
    ids_list = []
    scores_list = []
    for b in range(b):
        x = xs[k*b:k*(b+1)]
        for i in range(k):
            ids = np.unravel_index(x.argmax(), x.shape)
            score = x[ids[0]][ids[1]]
            ids_list.append((ids[0]+k*b, ids[1]))
            scores_list.append(score)
            for j in range(k):
                x[j][ids[1]] = -1000000
    return ids_list, scores_list


class AttentionEncoderDecoder(chainer.Chain):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            embed_size,
            hidden_size,
            atten_size):
        super(AttentionEncoderDecoder, self).__init__(
            # Encoder
            x_i = L.EmbedID(src_vocab_size, embed_size, ignore_label=-1),
            i_f = L.Linear(embed_size, 4 * hidden_size, nobias=True),
            f_f = L.Linear(hidden_size, 4 * hidden_size),
            i_b = L.Linear(embed_size, 4 * hidden_size, nobias=True),
            b_b = L.Linear(hidden_size, 4 * hidden_size),
            # Attention
            k_e = L.Linear(2 * hidden_size, atten_size, nobias=True),
            p_e = L.Linear(hidden_size, atten_size),
            e_a = L.Linear(atten_size, 1, nobias=True),
            # Decoder initializer
            fc_pc = L.Linear(hidden_size, hidden_size, nobias=True),
            bc_pc = L.Linear(hidden_size, hidden_size),
            f_p = L.Linear(hidden_size, hidden_size, nobias=True),
            b_p = L.Linear(hidden_size, hidden_size),
            # Decoder
            y_j = L.EmbedID(trg_vocab_size, embed_size, ignore_label=-1),
            j_p = L.Linear(embed_size, 4 * hidden_size, nobias=True),
            q_p = L.Linear(hidden_size, 4 * hidden_size, nobias=True),
            p_p = L.Linear(hidden_size, 4 * hidden_size),
            q_pq = L.Linear(hidden_size, hidden_size),
            p_pq = L.Linear(hidden_size, hidden_size),
            pq_z = L.Linear(hidden_size, trg_vocab_size),
            # add network
            fb_k = L.Linear(2 * hidden_size, hidden_size, nobias=True),
            d_l = L.Linear(hidden_size, hidden_size, nobias=True),
            kl_kle = L.Linear(hidden_size, hidden_size, nobias=True))
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.atten_size = atten_size


    def _encode(self, x_list):
        batch_size = len(x_list[0])
        source_length = len(x_list)

        # Encoding
        fc = bc = f = b = _zeros((batch_size, self.hidden_size))
        i_list = [self.x_i(_mkivar(x)) for x in x_list]
        f_list = []
        b_list = []
        for i in i_list:
            fc_tmp, f_tmp = F.lstm(fc, self.i_f(i) + self.f_f(f))
            enable = (i.data!=0)
            fc = F.where(enable, fc_tmp, fc)
            f = F.where(enable, f_tmp, f)
            f_list.append(f)
        for i in reversed(i_list):
            bc_tmp, b_tmp = F.lstm(bc, self.i_b(i) + self.b_b(b))
            enable = (i.data!=0)
            bc = F.where(enable, bc_tmp, bc)
            b = F.where(enable, b_tmp, b)
            b_list.append(b)
        b_list.reverse()

        # Making concatenated matrix
        # {f,b}_mat: shape = [batch, srclen, hidden]
        f_mat = F.concat([F.expand_dims(f, 1) for f in f_list], 1)
        b_mat = F.concat([F.expand_dims(b, 1) for b in b_list], 1)
        # fb_mat: shape = [batch, srclen, 2 * hidden]
        fb_mat = F.concat([f_mat, b_mat], 2)
        # k_mat: shape = [batch, srclen, hidden]
        k_mat = F.tanh(F.reshape(self.fb_k(F.reshape(fb_mat, [batch_size * source_length, 2 * self.hidden_size])), [batch_size, source_length, self.hidden_size]))
        # fbe_mat: shape = [batch * srclen, atten]
        #fbe_mat = self.fb_e(
        #    F.reshape(fb_mat, [batch_size * source_length, 2 * self.hidden_size]))

        return k_mat, fc, bc, f_list[-1], b_list[0]

    def _context(self, p, kl_mat, kle_mat):
        batch_size, all_length, _ = kl_mat.data.shape
        # {pe,e}_mat: shape = [batch * (srclen + trglen), atten]
        pe_mat = F.reshape(
            F.broadcast_to(
                F.expand_dims(self.p_e(p), 1),
                [batch_size, all_length, self.atten_size]),
            [batch_size * all_length, self.atten_size])
        e_mat = F.tanh(kle_mat + pe_mat)
        # a_mat: shape = [batch, (srclen + trglen)]
        a_mat = F.softmax(F.reshape(self.e_a(e_mat), [batch_size, all_length]))
        # q: shape = [batch, hidden]
        q = F.reshape(
            F.batch_matmul(a_mat, kl_mat, transa=True),
            [batch_size, self.hidden_size])

        return q

    def _initialize_decoder(self, fc, bc, f, b):
        return (
            F.tanh(self.fc_pc(fc) + self.bc_pc(bc)),
            F.tanh(self.f_p(f) + self.b_p(b)))

    def _decode_one_step(self, y, pc, p, q, kl_mat, kle_mat):
        j = self.y_j(_mkivar(y))
        pc, p = F.lstm(pc, self.j_p(j) + self.q_p(q) + self.p_p(p))
        q = self._context(p, kl_mat, kle_mat)
        pq = F.tanh(self.p_pq(p) + self.q_pq(q))
        z = self.pq_z(pq)
        return z, pc, p, q

    def _loss(self, z, t):
        return F.softmax_cross_entropy(z, _mkivar(t))

    def forward_train(self, x_list, t_list):
        batch_size = len(x_list[0])
        k_mat, fc, bc, f, b = self._encode(x_list)
        _, src_length, _ = k_mat.shape
        trg_length = len(t_list)
        pc, p = self._initialize_decoder(fc, bc, f, b)
        loss = _zeros(())
        q = _zeros((batch_size, self.hidden_size))
        d_mat = _zeros((batch_size, trg_length, self.hidden_size))
        for dec_num, (y, t) in enumerate(zip(t_list, t_list[1:])):
            # l_mat: shape = [batch, trg_len, hidden] 
            l_mat = F.reshape(F.tanh(self.d_l(F.reshape(d_mat, [batch_size*trg_length, self.hidden_size]))), [batch_size, trg_length, self.hidden_size])
            # kl_mat: shape = [batch, src+trg, hidden] 
            kl_mat = F.concat((k_mat, l_mat), 1)
            # kle_mat: shape = [batch*(src+trg), atten] 
            kle_mat = self.kl_kle(F.reshape(kl_mat, [batch_size * (src_length+trg_length), self.hidden_size]))
            z, pc, p, q = self._decode_one_step(y, pc, p, q, kl_mat, kle_mat)
            # update d_mat
            for b in range(batch_size):
                d_mat.data[b][dec_num] = p.data[b]
            loss += self._loss(z, t)
        return loss

    def forward_test(self, x_list, bos_id, eos_id, limit, beam_size):
        batch_size = len(x_list[0])
        k_mat, fc, bc, f, b = self._encode(x_list)
        _, src_length, _ = k_mat.shape
        trg_length = len(t_list)
        pc, p = self._initialize_decoder(fc, bc, f, b)
        z_list = []
        y = [bos_id for _ in range(batch_size)]
        q = _zeros((batch_size, 2 * self.hidden_size))
        while True:
            z, pc, p, q = self._decode_one_step(y, pc, p, q, fb_mat, fbe_mat)
            z = [int(w) for w in z.data.argmax(1)]
            z_list.append(z)
            if all(w == eos_id for w in z):
                break
            elif len(z_list) >= limit:
                z_list.append([eos_id for _ in range(batch_size)])
                break
            y = z
        return z_list
