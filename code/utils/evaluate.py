import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(hid_units, test_embs, test_lbls, val_embs, val_lbls, train_embs, train_lbls, nb_classes, device,
    lr, wd, isTest=True):
    # hid_units = test_embs.shape[1]
    xent = nn.CrossEntropyLoss()
    train_lbls = torch.argmax(train_lbls, dim=-1)
    # train_lbls = train_lbls.reshape(1, 1)
    test_lbls = torch.argmax(test_lbls,dim=-1)
    # test_lbls = test_lbls.reshape(1, 1)
    val_lbls = torch.argmax(val_lbls,dim=-1)
    val_lbls = val_lbls.reshape(1)

    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_micro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()  # 将逻辑回归模型设置为训练模式
            opt.zero_grad()  # 将优化器的梯度缓冲区清零

            logits = log(train_embs)  # 使用逻辑回归模型进行训练样本的预测，得到预测的logits
            loss = xent(logits, train_lbls)  # 计算预测logits和训练样本标签之间的交叉熵损失

            loss.backward()  # 反向传播，计算梯度
            opt.step()  # 根据计算得到的梯度更新逻辑回归模型的参数

            logits = log(val_embs)  # 使用逻辑回归模型对验证集样本进行预测，得到预测的logits
            # logits = logits.reshape(logits.shape[0],1)
            logits = logits.reshape(1,logits.shape[0])
            # print(logits.shape)
            preds = torch.argmax(logits, dim=1)  # 根据logits计算预测的类别
            preds = preds.reshape(preds.shape[0])
            # print(val_lbls.shape)
            # print(preds.shape)
            # exit(0)
            # val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')
            # val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            logits = log(test_embs)
            logits_list.append(logits)

        max_iter = val_micro_f1s.index(max(val_micro_f1s))

        # auc
        best_logits = logits_list[max_iter]
        best_logits = best_logits.reshape(best_logits.shape[0],1)
        best_proba = softmax(best_logits, dim=1)
        # print(test_lbls)
        # print(best_proba)
        test_lbls = test_lbls.reshape(1,1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),

                                            ))

    if isTest:
        print("\t[Classification] auc {:.4f} var {:.4f}"
              .format(np.mean(auc_score_list),  # 平均AUC
                      np.std(auc_score_list)  # AUC方差
                      )
              )
        # return np.mean(auc_score_list)
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)
