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


def length_penalty(length, alpha=0.6):
    lp = float(((5 + length)**alpha) / ((5 + 1)**alpha))
    return lp


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
            fb_e = L.Linear(2 * hidden_size, atten_size, nobias=True),
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
            q_p = L.Linear(2 * hidden_size, 4 * hidden_size, nobias=True),
            p_p = L.Linear(hidden_size, 4 * hidden_size),
            q_pq = L.Linear(2 * hidden_size, hidden_size),
            p_pq = L.Linear(hidden_size, hidden_size),
            pq_z = L.Linear(hidden_size, trg_vocab_size))
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
            in_fwlstm = F.dropout(self.i_f(i) + self.f_f(f), ratio=0.3)
            fc_tmp, f_tmp = F.lstm(fc, in_fwlstm)
            f_tmp = F.dropout(f_tmp, ratio=0.3)
            enable = (i.data!=0)
            fc = F.where(enable, fc_tmp, fc)
            f = F.where(enable, f_tmp, f)
            f_list.append(f)
        for i in reversed(i_list):
            in_bwlstm = F.dropout(self.i_b(i) + self.b_b(b), ratio=0.3)
            bc_tmp, b_tmp = F.lstm(bc, in_bwlstm)
            b_tmp = F.dropout(b_tmp, ratio=0.3)
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
        # fbe_mat: shape = [batch * srclen, atten]
        fbe_mat = self.fb_e(
            F.reshape(fb_mat, [batch_size * source_length, 2 * self.hidden_size]))

        return fb_mat, fbe_mat, fc, bc, f_list[-1], b_list[0]

    def _context(self, p, fb_mat, fbe_mat):
        batch_size, source_length, _ = fb_mat.data.shape
        # {pe,e}_mat: shape = [batch * srclen, atten]
        pe_mat = F.reshape(
            F.broadcast_to(
                F.expand_dims(self.p_e(p), 1),
                [batch_size, source_length, self.atten_size]),
            [batch_size * source_length, self.atten_size])
        e_mat = F.tanh(fbe_mat + pe_mat)
        # a_mat: shape = [batch, srclen]
        a_mat = F.softmax(F.reshape(self.e_a(e_mat), [batch_size, source_length]))
        # q: shape = [batch, 2 * hidden]
        q = F.reshape(
            F.batch_matmul(a_mat, fb_mat, transa=True),
            [batch_size, 2 * self.hidden_size])

        return q

    def _initialize_decoder(self, fc, bc, f, b):
        return (
            F.tanh(self.fc_pc(fc) + self.bc_pc(bc)),
            F.tanh(self.f_p(f) + self.b_p(b)))

    def _decode_one_step(self, y, pc, p, q, fb_mat, fbe_mat):
        j = self.y_j(_mkivar(y))
        in_lstm = F.dropout(self.j_p(j) + self.q_p(q) + self.p_p(p), ratio=0.3)
        pc, p = F.lstm(pc, in_lstm)
        p = F.dropout(p, ratio=0.3)
        q = self._context(p, fb_mat, fbe_mat)
        pq = F.tanh(self.p_pq(p) + self.q_pq(q))
        z = self.pq_z(pq)
        return z, pc, p, q

    def _loss(self, z, t):
        return F.softmax_cross_entropy(z, _mkivar(t))

    def forward_train(self, x_list, t_list):
        all_size = len(x_list[0])
        fb_mat, fbe_mat, fc, bc, f, b = self._encode(x_list)
        pc, p = self._initialize_decoder(fc, bc, f, b)
        loss = _zeros(())
        q = _zeros((all_size, 2 * self.hidden_size))
        for y, t in zip(t_list, t_list[1:]):
            z, pc, p, q = self._decode_one_step(y, pc, p, q, fb_mat, fbe_mat)
            loss += self._loss(z, t)
        return loss

    def forward_test(self, x_list, bos_id, eos_id, limit, beam_size):
        all_size = len(x_list[0])
        batch_size = all_size / beam_size
        fb_mat, fbe_mat, fc, bc, f, b = self._encode(x_list)
        pc, p = self._initialize_decoder(fc, bc, f, b)
        z_list = []
        y = [bos_id for _ in range(all_size)]
        q = _zeros((all_size, 2 * self.hidden_size))
        z_tmp = [0 for _ in range(all_size)]
        eos_flag = [False for _ in range(all_size)]
        all_z = [[] for _ in range(all_size)]
        flag = 0
        count = 1
        while True:
            # z's shape: (all_size, trg_vocab_size)
            # pc's shape: (all_szie, hidden_size)
            # p's shape: (all_size, hidden_size)
            # q's shape: (all_size, 2 * hidden_size)
            z, pc, p, q = self._decode_one_step(y, pc, p, q, fb_mat, fbe_mat)
            # 累積対数尤度の計算
            z = chainer.cuda.to_cpu(F.log_softmax(z).data)
            for i in range(all_size):
                if count > 1:
                    if eos_flag[i]:
                        z[i,:] = z_tmp[i]
                    else:
                        z[i,:] += (z_tmp[i] * length_penalty((count - 1)))
                        z[i,:] = z[i,:] / float(length_penalty(count))
            
            # top-kを計算
            if flag == 0:
                ids_list, scores_list = top_k_init(z, beam_size, batch_size)
                flag = 1
            else:
                ids_list, scores_list = top_k(z, beam_size, batch_size)
            # 隠れ層状態の更新
            y = []
            pc_tmp = copy.deepcopy(pc.data)
            p_tmp = copy.deepcopy(p.data)
            q_tmp = copy.deepcopy(q.data)
            all_z = [all_z[ids[0]] + [ids[1]] for ids in ids_list]

            for i, (ids, scores) in enumerate(zip(ids_list, scores_list)):
                pc.data[i] = copy.deepcopy(pc_tmp[ids[0]])
                p.data[i] = copy.deepcopy(p_tmp[ids[0]])
                q.data[i] = copy.deepcopy(q_tmp[ids[0]])
                z_tmp[i] = scores
                y.append(ids[1])
                if ids[1] == eos_id:
                    eos_flag[i] = True
            if all(ids[1] == eos_id for ids in ids_list):
                result = []
                for i in range(0, all_size, beam_size):
                    result.append(all_z[i])
                break
            elif len(all_z[0]) >= limit:
                result = []
                for i in range(0, all_size, beam_size):
                    all_z[i].append(eos_id)
                    result.append(all_z[i])
                break
            count += 1
        return result

    def forward_test_orig(self, x_list, bos_id, eos_id, limit):
        batch_size = len(x_list[0])
        fb_mat, fbe_mat, fc, bc, f, b = self._encode(x_list)
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
