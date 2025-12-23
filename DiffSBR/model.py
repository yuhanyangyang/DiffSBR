import datetime
import math
import numpy as np
import torch
import random
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
import pickle
import os
import torch
import torch.nn as nn

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, emb_size, n_node):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.n_node = n_node
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.w_co = nn.Linear(self.emb_size, self.emb_size)

        self.gate_w1 = nn.Linear(self.emb_size * 2, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency, embedding, mo_embedding):
        id_embeddings = self.dropout30(embedding)
        mo_embedding = self.dropout30(mo_embedding)

        for i in range(self.layers):
            item_co_emb = self.get_embedding(adjacency, id_embeddings)
            id_embeddings = id_embeddings + item_co_emb
            item_mo_emb = self.get_embedding(adjacency, mo_embedding)
            mo_embedding = mo_embedding + item_mo_emb
        results = id_embeddings
        mo_results = mo_embedding
        return results, mo_results

    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embedding)
        adjacency = trans_to_cuda(adjacency)
        # item_embeddings = torch.sparse.mm(adjacency, trans_to_cuda(embedding))
        item_embeddings = torch.mm(adjacency.to_dense(), trans_to_cuda(embedding))
        return item_embeddings


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Diff(Module):
    def __init__(self, opt, n_node, adjacency, lr, co_layer, l2, item_beta, pro_beta, cri_beta, dataset, num_heads=4,
                 # emb_size=100, text_emb_size=100, img_emb_size=100, batch_size=100, num_negatives=100):
                 emb_size=100, text_emb_size=100, batch_size=100, num_negatives=100):
        super(Diff, self).__init__()
        self.emb_size = emb_size
        self.text_emb_size = text_emb_size
        # self.img_emb_size = img_emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.co_layer = co_layer
        self.item_beta = item_beta
        self.pro_beta = pro_beta
        self.cri_beta = cri_beta
        self.num_negatives = num_negatives
        self.num_heads = num_heads
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.FindNeighbor = FindSimilarIntentSess(self.emb_size)
        self.LearnableRetriever = LearnableRetriever(self.emb_size,neighbor_n=3, temperature=0.5)
        self.match_loss = MatchLoss(temperature=0.3)
        self.n_target = 3
        self.w_ne = 1.7
        self.w = 20
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.adjacency = adjacency
        self.HyperGraph = HyperConv(self.co_layer, self.emb_size, self.n_node)
        text_path = './datasets/' + dataset + '/textMatrixpca100.npy'
        textWeights = np.load(text_path)
        self.text_embedding = nn.Embedding(self.n_node, text_emb_size)
        text_pre_weight = np.array(textWeights)
        self.text_embedding.weight.data.copy_(torch.from_numpy(text_pre_weight))
        self.pos_embedding = nn.Embedding(2000, self.emb_size)
        self.active = nn.ReLU()
        self.relu = nn.ReLU()

        if self.emb_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))

        self.attention_head_size = int(emb_size / num_heads)
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        self.query = nn.Linear(self.emb_size, self.emb_size)
        self.key = nn.Linear(self.emb_size, self.emb_size)
        self.value = nn.Linear(self.emb_size, self.emb_size)

        self.query_id = nn.Linear(self.emb_size, self.emb_size)
        self.key_id = nn.Linear(self.emb_size, self.emb_size)
        self.value_id = nn.Linear(self.emb_size, self.emb_size)

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)
        self.embed_dropout = nn.Dropout(0.3)
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)

        self.cos_sim = nn.CosineSimilarity(dim=-1)

        self.sim_w_p1u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p1d = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p2u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p2d = nn.Linear(self.emb_size, self.emb_size)

        self.sim_w_c1u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c2u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c1d = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c2d = nn.Linear(self.emb_size, self.emb_size)

        self.sim_w_i1 = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_i2 = nn.Linear(self.emb_size, self.emb_size)

        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.mlp_seq1 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq2 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq3 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq4 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq5 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq6 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq7 = nn.Linear(self.emb_size, self.emb_size)
        self.mlp_seq8 = nn.Linear(self.emb_size, self.emb_size)

        self.linear_one = nn.Linear(self.emb_size, self.emb_size)
        self.linear_two = nn.Linear(self.emb_size, self.emb_size)
        self.linear_three = nn.Linear(self.emb_size, self.emb_size)
        self.linear_transform = nn.Linear(2 * self.emb_size, self.emb_size)
        self.linear_one1 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_two1 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_three1 = nn.Linear(self.emb_size, self.emb_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, text_embedding, session_item, session_len,
                          reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        mask = mask.float().unsqueeze(-1)
        item_emb_table = torch.cat([zeros, item_embedding], 0)
        text_emb_table = torch.cat([zeros, text_embedding], 0)
        get = lambda i: item_emb_table[session_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        mask = (session_item != 0).int()
        ht = seq_h[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(seq_h)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        sess_embedding = torch.sum(alpha * seq_h * mask.view(mask.shape[0], -1, 1).float(), 1)
        id_sess_pro_emb = torch.div(torch.sum(seq_h, 1), session_len.type(torch.cuda.FloatTensor))
        get_text = lambda i: text_emb_table[session_item[i]]
        seq_text = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_text[i] = get_text(i)
        mo_sess_pro_emb = torch.div(torch.sum(seq_text, 1), session_len.type(torch.cuda.FloatTensor))

        ht1 = seq_text[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q11 = self.linear_one1(ht1).view(ht1.shape[0], 1, ht.shape[1])
        q21 = self.linear_two1(seq_text)
        alpha1 = self.linear_three1(torch.sigmoid(q11 + q21))
        sess_embedding_mo = torch.sum(alpha1 * seq_text * mask.view(mask.shape[0], -1, 1).float(), 1)
        return sess_embedding, sess_embedding_mo, id_sess_pro_emb, mo_sess_pro_emb


    def sim_cal_w(self, emb1, emb2, mat_w):
        sim = emb1 * mat_w(emb2)
        sim = torch.sum(sim, -1)
        return sim

    def transpose_for_scores(self, x, attention_head_size):
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, session_item, flag, session_len, reversed_sess_item, mask, target):

        item_emb = trans_to_cuda(self.embedding.weight)
        # item_emb = self.embed_dropout(item_emb)
        text_emb = trans_to_cuda(self.text_embedding.weight)
        # text_emb = self.embed_dropout(text_emb)
        item_emb, text_emb= self.HyperGraph(self.adjacency, item_emb, text_emb)
        inter_loss = self.match_loss(item_emb, text_emb)
        # inter_loss = 0
        sess_id_emb, sess_mo_emb,  sess_id_pro_emb, sess_mo_pro_emb = self.generate_sess_emb(item_emb,text_emb,session_item, session_len, reversed_sess_item,mask)
        sess_topk, neighbor_sess,cos_topk= self.LearnableRetriever(sess_id_emb,sess_id_emb)
        return item_emb, text_emb, sess_id_emb, sess_mo_emb, neighbor_sess,sess_id_pro_emb, sess_mo_pro_emb,cos_topk, sess_topk,inter_loss

class LearnableRetriever(Module):
    def __init__(self, hidden_size, neighbor_n=3, temperature=0.5):
        super(LearnableRetriever, self).__init__()
        self.hidden_size = hidden_size
        self.neighbor_n = neighbor_n
        self.temperature = temperature
        self.dropout40 = nn.Dropout(0.40)

        self.scoring = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.sim_weight = nn.Parameter(torch.ones(self.neighbor_n), requires_grad=True)

    def compute_sim(self, sess_emb, pool_emb):
        B, D = sess_emb.size()
        N = pool_emb.size(0)
        sess_expand = sess_emb.unsqueeze(1).expand(B, N, D)
        pool_expand = pool_emb.unsqueeze(0).expand(B, N, D)
        concat_feat = torch.cat([sess_expand, pool_expand], dim=-1)
        sim_score = self.scoring(concat_feat).squeeze(-1)
        return sim_score

    def forward(self, sess_emb, pool_emb):
        sess_proj = self.scoring(sess_emb)
        sim_matrix = torch.matmul(sess_proj, sess_proj.T)
        cos_topk, topk_indice = torch.topk(sim_matrix, self.neighbor_n, dim=1)
        sess_topk = sess_emb[topk_indice]
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        weight = cos_topk.unsqueeze(2).expand_as(sess_topk)
        neighbor_sess = torch.sum(weight * sess_topk, dim=1)
        return sess_topk, neighbor_sess,cos_topk
#



class FindSimilarIntentSess(nn.Module):
    def __init__(self, hidden_size):
        super(FindSimilarIntentSess, self).__init__()
        self.hidden_size = hidden_size
        self.neighbor_n = 3
        self.dropout40 = nn.Dropout(0.40)

    def compute_sim(self, sess_emb, pool_emb):

        fenzi = torch.matmul(sess_emb, pool_emb.permute(1, 0))
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, dim=1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu_r = torch.sum(pool_emb * pool_emb + 0.000001, dim=1)
        fenmu_r = torch.sqrt(fenmu_r).unsqueeze(0)
        fenmu = torch.matmul(fenmu_l, fenmu_r)
        cos_sim = fenzi / fenmu
        cos_sim = nn.Softmax(dim=-1)(cos_sim)
        return cos_sim


    def forward(self, sess_emb, pool_emb):
        k_v = self.neighbor_n
        cos_sim = self.compute_sim(sess_emb, pool_emb)
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = pool_emb[topk_indice]
        cos_sim = cos_topk.unsqueeze(2).expand(
            cos_topk.size()[0], cos_topk.size()[1], self.hidden_size
        )
        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)
        return neighbor_sess, cos_topk, sess_topk

class MatchLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super().__init__()
        self.T = temperature

    def forward(self, feature_left, feature_right):
        id_features = F.normalize(feature_left, dim=1)
        modal_features = F.normalize(feature_right, dim=1)
        similarity = torch.mm(id_features, modal_features.t()) / self.T
        exp_similarity = torch.exp(similarity)
        positive_sim = torch.diag(exp_similarity)
        denominator = exp_similarity.sum(dim=1)
        loss = -torch.log(positive_sim / denominator)
        return loss.mean()

def perform(model, diffusion, id_reverse_model,mo_reverse_model, i, data):
    tar, flag, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    flag = trans_to_cuda(torch.Tensor(flag).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    # item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb, con_loss = model(session_item, flag, session_len, reversed_sess_item, mask, tar)
    item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb, neighbor_sess,sess_id_pro_emb,sess_mo_pro_emb,cos_topk, sess_topk,inter_loss  = model(session_item, flag, session_len, reversed_sess_item, mask,tar)
    # diff_neig_result = diffusion.training_losses(item_reverse_model, neighbor_sess, reweight=True)
    diff_id_neig_result, diff_mo_result, sess_id_pro_emb1 = diffusion.training_losses1(id_reverse_model, mo_reverse_model, sess_id_emb, sess_mo_emb, sess_topk, reweight=True)
    id_diff_loss = diffusion.get_reconstruct_loss(sess_id_pro_emb1, diff_id_neig_result)
    id_diff_loss = torch.mean(torch.sum(cos_topk * id_diff_loss, dim=1))
    mo_diff_loss = diffusion.get_reconstruct_loss(sess_id_pro_emb1, diff_mo_result)
    mo_diff_loss = torch.mean(mo_diff_loss)
    diff_id_neig_result = diff_id_neig_result.mean(dim=1)
    diff_mo_result = diff_mo_result.mean(dim=1)
    match_loss1 = MatchLoss(temperature=0.3)
    conloss = match_loss1(diff_id_neig_result,diff_mo_result)
    sess_id_emb = 0.5 * sess_id_emb+ 0.5 * diff_id_neig_result
    scores_id = torch.mm(sess_id_emb, torch.transpose(item_id_emb, 1, 0))
    scores = scores_id
    neigh_conloss = inter_loss
    diff_loss = id_diff_loss + mo_diff_loss
    return tar, scores, diff_loss,conloss,neigh_conloss


def perform_infer(model, diffusion, id_reverse_model,mo_reverse_model, i, data):
    tar, flag, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    flag = trans_to_cuda(torch.Tensor(flag).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb, neighbor_sess, sess_id_pro_emb, sess_mo_pro_emb, cos_topk, sess_topk,inter_loss = model(session_item, flag, session_len, reversed_sess_item, mask, tar)
    sess_id_emb1 = sess_id_emb.unsqueeze(1).repeat(1, 3, 1)
    diff_id_neig_result = diffusion.p_sample(id_reverse_model, sess_id_emb1, sess_topk, 0, False)
    diff_id_neig_result = diff_id_neig_result.mean(dim=1)
    sess_id_emb = 0.9*sess_id_emb + 0.1*diff_id_neig_result
    scores_id = torch.mm(sess_id_emb, torch.transpose(item_id_emb, 1, 0))
    scores = scores_id
    return tar, scores, sess_id_emb

def train_test(model, diffusion,id_reverse_model,mo_reverse_model, train_data, test_data, epoch):
    model.train()
    id_reverse_model.train()
    mo_reverse_model.train()
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        id_reverse_model.optimizer.zero_grad()
        mo_reverse_model.optimizer.zero_grad()
        targets, scores, diff_loss,conloss,contrastive_loss = perform(model, diffusion, id_reverse_model,mo_reverse_model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + diff_loss*8 + conloss*0.05 + contrastive_loss
        loss.backward()
        model.optimizer.step()
        id_reverse_model.optimizer.step()
        mo_reverse_model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    id_reverse_model.eval()
    mo_reverse_model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores, sess_id = perform_infer(model,diffusion, id_reverse_model,mo_reverse_model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class DNN(nn.Module):

    def __init__(self,opt, hidden_size,in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5,lr=0.001):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.hidden_size = hidden_size
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4), SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size))

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0]*2 + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        self.in_layers = nn.ModuleList([nn.Linear(300, 200)])
        self.out_layers = nn.ModuleList([nn.Linear(200, 100)])

        self.drop = nn.Dropout(dropout)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.init_parameters()
        self.init_weights()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.time_emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, neighbor_sess_mo, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, x.size(-1)).to(x.device))
        emb = emb.unsqueeze(1).repeat(1, x.size(1), 1)  # (50, 5, 10)

        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.drop(x)
        h = torch.cat([x, emb, neighbor_sess_mo], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h



def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DNN1(nn.Module):

    def __init__(self,opt, hidden_size,in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5,lr=0.001):
        super(DNN1, self).__init__()
        self.in_dims = in_dims
        self.hidden_size = hidden_size
        self.out_dims = out_dims
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4), SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size))

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        self.in_layers = nn.ModuleList([nn.Linear(200, 150)])
        self.out_layers = nn.ModuleList([nn.Linear(150, 100)])

        self.drop = nn.Dropout(dropout)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.init_parameters()
        self.init_weights()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.time_emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:

            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, x.size(-1)).to(x.device))
        emb = emb.unsqueeze(1).repeat(1, x.size(1), 1)
        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h

class DNN2(nn.Module):

    def __init__(self,opt, hidden_size,in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5,lr=0.001):
        super(DNN2, self).__init__()
        self.in_dims = in_dims
        self.hidden_size = hidden_size
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4), SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size))

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        self.in_layers = nn.ModuleList([nn.Linear(200, 150)])
        self.out_layers = nn.ModuleList([nn.Linear(150, 100)])
        self.drop = nn.Dropout(dropout)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.init_parameters()
        self.init_weights()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.time_emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, x.size(-1)).to(x.device))
        emb = emb.unsqueeze(1).repeat(1, x.size(1), 1)
        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h
