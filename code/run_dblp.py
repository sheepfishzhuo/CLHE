import numpy
import torch
from utils import set_params, evaluate
from module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random
import numpy as np
from scipy import sparse
import scipy
import torch as th
from scipy import sparse
import scipy.sparse as sp
import dgl
from util.tools import evaluate_results_nc
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
args = set_params('DBLP')
if torch.cuda.is_available():
    device = torch.device("cpu")
    # torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")
# torch.device('cpu')
## name of in
# termediate document ##
own_str = 'DBLP'

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

data_path = '../data/DBLP_L/'
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()



def load_data1():
    apa = sp.load_npz(data_path + "apa.npz")
    apcpa = sp.load_npz(data_path + "apvpa.npz")
    aptpa = sp.load_npz(data_path + "aptpa.npz")
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))

    feature =scipy.sparse.load_npz(data_path + 'a_feature.npz').toarray()
    # feature =scipy.sparse.load_npz(data_path + 'a_feature.npz').astype("float32")
    # feature = th.FloatTensor(preprocess_features(feature))

    label=np.zeros((4057,4))
    idx_train = []
    idx_val = []
    idx_test= []
    f3 = open(data_path + 'label_train.dat', 'r', encoding='utf-8')
    for line in f3.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        a = int(a)
        c = int(c)
        label[a][c] = 1
        idx_train.append(a)
    f4 = open(data_path + 'label_val.dat', 'r', encoding='utf-8')
    for line in f4.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        a = int(a)
        c = int(c)
        label[a][c] = 1
        idx_val.append(a)
    f5 = open(data_path + 'label_test.dat', 'r', encoding='utf-8')
    for line in f5.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        a = int(a)
        c = int(c)
        label[a][c] = 1
        idx_test.append(a)

    path = "../data/DBLP_L/"
    pos = sp.load_npz(path + "pos_dblp_apcpa_aptpa700.npz")
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    mps = [apa, apcpa, aptpa]
    return feature, label, idx_train, idx_val, idx_test,pos, mps
# features, labels, idx_train, idx_val, idx_test,pos = load_data1()

def get_A(data_path):
    # adj=scipy.sparse.load_npz(data_path + 'adjM.npz').toarray()
    # pa = np.zeros((14328,4057))
    # pt = np.zeros((14328, 7723))
    # pv = np.zeros((14328, 20))
    # for i in range(4057,4057+14328):
    #     for j in range(len(adj)):
    #         if adj[i][j]!=0:
    #             if j<4057:
    #                 pa[i-4057][j]=1
    #             elif j>=(4057+14328) and j<(4057+14328+7723):
    #                 pt[i-4057][j-4057-14328]=1
    #             elif j>=(4057+14328+7723):
    #                 pv[i-4057][j-4057-14328-7723]=1
    # ap=pa.T
    # tp=pt.T
    # vp=pv.T
    # apa=ap.dot(pa)
    # sp.save_npz(data_path + 'apa.npz', scipy.sparse.csr_matrix(apa))
    # apt = ap.dot(pt)
    # aptp = apt.dot(tp)
    # aptpa = aptp.dot(pa)
    # sp.save_npz(data_path + 'aptpa.npz', scipy.sparse.csr_matrix(aptpa))
    # apv = ap.dot(pv)
    # apvp = apv.dot(vp)
    # apvpa = apvp.dot(pa)
    # sp.save_npz(data_path + 'apvpa.npz', scipy.sparse.csr_matrix(apvpa))
    apa = scipy.sparse.load_npz(data_path + 'apa.npz').toarray()
    print('apa_max:',apa[np.unravel_index(np.argmax(apa, axis=None), apa.shape)])
    apvpa = scipy.sparse.load_npz(data_path + 'apvpa.npz').toarray()
    print('apvpa_max:', apa[np.unravel_index(np.argmax(apvpa, axis=None), apvpa.shape)])
    aptpa = scipy.sparse.load_npz(data_path + 'aptpa.npz').toarray()
    print('aptpa_max:', apa[np.unravel_index(np.argmax(aptpa, axis=None), aptpa.shape)])
    aa=4*apvpa+1*apa+aptpa#+aptpa
    print('aa_max:', apa[np.unravel_index(np.argmax(apa, axis=None), apa.shape)])
    return aa

def normalize(mx):
    """Row-normalize sparse matrix   """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def train():
    features, labels, idx_train, idx_val, idx_test, pos, mps = load_data1()
    # print(features.shape)  (4057, 334)
    feats_dim =features.shape[1]
    features = torch.from_numpy(features)
    features = features.float()

    P = int(len(mps))
    '''
    print(features)
    tensor([[0., 0., 1.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
    '''
    # print(features.shape)   torch.Size([4057, 334])
    # exit()
    labels = torch.LongTensor(np.where(labels)[1])
    # Y = labels.cpu().numpy()
    '''
    print(Y)  [2 2 3 ... 0 0 0]
    print(Y.shape)  (4057,)
    print(len(Y))  4057
    '''



    adj= get_A(data_path) # adj作为GCN下的矩阵  调用函数get_A(data_path)获取矩阵adj，在后续GCN模型中使用
    adj[adj <40  ] = 0# GCN去掉条边的链接,全部的话改成13，apa和apvpa的话改成7  将矩阵adj中小于40的值设为0
    adj = torch.from_numpy(adj).type(torch.FloatTensor)  # 将矩阵adj转换为PyTorch的浮点型张量
    adj = F.normalize(adj, dim=1, p=2)  # 对adj进行归一化处理
    adj = scipy.sparse.csr_matrix(adj)  # 将归一化后的adj转换为CSR格式的稀疏矩阵
    # e是边
    e = torch.tensor(adj.data).type(torch.FloatTensor)   # 从稀疏矩阵adj中获取非零元素，并转换为PyTorch的浮点型张量
    g1 = dgl.DGLGraph(adj)  # 这里是GCN的   创建了一个DGL图g1，用于GCN模型

    # A = get_A(data_path) # A为GAT下的矩阵  调用函数get_A(data_path)获取矩阵adj，在后续GAT模型中使用
    # A[A <40] = 0#GAT去掉6条边的链接,全部的话改成13，apa和apvpa的话改成7
    # A = torch.from_numpy(A).type(torch.FloatTensor)
    # A = normalize(A)  # 对矩阵A进行归一化处理
    # A = np.array(A.tolist())  # 将归一化后的矩阵A转换为numpy数组
    # adjM2 = scipy.sparse.csr_matrix(A)  # 将numpy数组A转换为CSR格式的稀疏矩阵
    #
    # g = dgl.DGLGraph(adjM2)  # 创建了一个DGL图g，用于GAT模型
    # g = dgl.remove_self_loop(g)  # 移除图g中的自环
    # g = dgl.add_self_loop(g)  # 这里的是GAT的  为图g添加自环

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    model = HeCo(args.hidden_dim, feats_dim, args.feat_drop, args.attn_drop,
                     args.tau, args.lam, P)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        # loss = model(features, g1,e,g,pos)
        loss = model(features, g1, e, mps,pos)
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HeCo_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HeCo_'+own_str+'.pkl'))
    model.eval()
    os.remove('HeCo_'+own_str+'.pkl')
    # embeds = model.get_embeds(features, g1,e,g)#这里我们返回的是GAT路径下的结果
    # embeds = model.get_embeds(features, g1, e, g)  # 这里我们返回的是GAT路径下的结果

    embeds = model.get_embeds(features, g1, e, mps)  # 这里我们返回的是GAT路径下的结果

    # #=============================以下代码为可视化的效果
    # Y = labels.cpu().numpy()
    # ml = TSNE(n_components=2)
    # node_pos = ml.fit_transform(embeds.cpu().data.numpy())
    # color_idx = {}
    # for i in range(4057):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():  # c是类型数，idx是索引
    #     if str(c) == '1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#48D1CC', s=15, alpha=1)
    #     elif str(c) == '2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#FFD700', s=15, alpha=1)
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#7B68EE', s=15, alpha=1)
    #     elif str(c) == '3':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#CD5C5C', s=15, alpha=1)
    # plt.legend()
    # plt.savefig("hetgnn-sf-DBLP可视化" +  ".png", dpi=1000, bbox_inches='tight')
    # plt.show()
    # # =======================================================一直到这里↑↑都是可视化
    #进行测试
    # nb_classes = labels.shape[-1]
    # for i in range(len(idx_train)):    # 对训练集评估（评价指标）
    #     evaluate(embeds.shape[1], embeds[idx_test[i]].cpu(), labels[idx_test[i]].cpu(), embeds[idx_val[i]].cpu(), labels[idx_val[i]].cpu(), embeds[idx_train[i]].cpu(), labels[idx_train[i]].cpu(),nb_classes, device, args.eva_lr, args.eva_wd)  # 输出ma mi AUC值
    # evaluate(embeds.shape[1], embeds[idx_test].cpu(), labels[idx_test].cpu(), embeds[idx_val].cpu(), labels[idx_val].cpu(), embeds[idx_train].cpu(), labels[idx_train].cpu(),nb_classes, device, args.eva_lr, args.eva_wd)  # 输出ma mi AUC值
    svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeds[idx_test].cpu().data.numpy(), labels[idx_test].cpu().numpy(), 3)
    #保存嵌入
    f = open("./embeds/" + args.dataset + str(args.turn) + ".pkl", "wb")
    pkl.dump(embeds.cpu().data.numpy(), f)
    f.close()

if __name__ == '__main__':
    train()
