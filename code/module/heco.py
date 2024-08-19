import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .contrast import Contrast

'''

'''
class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim, feat_drop, attn_dropout, tau, lam, P):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim # 64
        # self.fc_list = nn.Linear(feats_dim, feats_dim, bias=True)
        self.fc_list = [nn.Linear(feats_dim, hidden_dim, bias=True), nn.Linear(feats_dim, feats_dim, bias=True)]
        self.mp = Mp_encoder(feats_dim, hidden_dim,1, feat_drop)
        # self.sc = Sc_encoder(feats_dim, hidden_dim, feat_drop, attn_dropout)
        self.sc = Sc_encoder(P, hidden_dim, attn_dropout)
        self.contrast = Contrast(hidden_dim, tau, lam)


        nn.init.xavier_normal_(self.fc_list[0].weight, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

    def forward(self, x,adj,e,A,pos):
        h_all = self.fc_list[0](x)
        x = self.fc_list[1](x)
        # h_all = F.elu(self.feat_drop(x))
        z_mp = self.mp(x,adj,e)  # GCN
        # z_sc = self.sc(x,A)   # GAT
        z_sc = self.sc(h_all, A)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, x,adj,e,A):
        x = self.fc_list[0](x)
        z_sc = self.sc(x,A)  # GAT
        return z_sc.detach()





'''
class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim, feat_drop, attn_dropout, tau, lam):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim # 64
        self.fc_list = nn.Linear(feats_dim, feats_dim, bias=True)
        self.mp = Mp_encoder(feats_dim, hidden_dim,1, feat_drop)
        self.sc = Sc_encoder(feats_dim, hidden_dim, feat_drop, attn_dropout)
        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, x,adj,e,A,pos):
        x = self.fc_list(x)
        z_mp = self.mp(x,adj,e)  # GCN
        z_sc = self.sc(x,A)   # GAT
        loss = self.contrast(z_mp, z_sc, pos)
        return loss


    def get_embeds(self, x,adj,e,A):
        x = self.fc_list(x)
        # z_mp = self.mp(x,adj,e)
        # return z_mp.detach()
        z_sc = self.sc(x,A)  # GAT
        return z_sc.detach()
'''

