#encoding:utf-8

import numpy as np

# 读npz
# 加载npz文件
data = np.load("D:/Study/codes/embedding/HetGNN-SF--main/HetGNN-SF--main/data/DBLP_L/train_val_test_idx.npz")

# 获取所有键列表
keys = data.files
print(keys)
for key in data:
    arr = data[key]
    print(f"数组名称: {key}, 形状: {arr.shape}")


# 遍历键并访问对应的值
# for key in keys:
#     value = data[key]
#     print(key, value)
# val_idx = data['val_idx']
# np.savetxt('D:/Study/codes/embedding/HeCo-main0805/HeCo-main/data/dblp/a_feat.txt', val_idx,fmt='%d')




# # 读npy
# # 加载npy文件
# data = np.load("D:/Study/codes/embedding/HeCo-main/HeCo-main/data/dblp/test_20.npy",allow_pickle = True)
#
# # 打印数组内容
# print(data)
# print(len(data))