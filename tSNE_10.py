# from tsnecuda import TSNE
from tkinter import Y
from tkinter.tix import X_REGION, Tree
from sklearn.manifold import TSNE
import torch
import numpy as np
import einops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import tqdm
import os
# import tsnecuda



X = torch.load('/data/lzj/easy/save_tensor/raw_feat/proto_feat_1.pkl')

Y = np.arange(0, 64)


rootpath = '/data/lzj/easy/save_fig'
if not os.path.exists(rootpath):
    os.makedirs(rootpath)

def plot_embedding(X, Y, save_path):
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
 
    fig, ax = plt.subplots()
    print("done2")
    # plt.scatter(X[:,0], X[:,1])
    # for i in range(X.shape[0]): 
    #     plt.text(X[i, 0], X[i, 1], str(Y[i]),
    #              color=plt.cm.Set1(Y[i] / 100.),  #cm代表color map，即颜色映射地图，Set1, Set2, Set3是它的三个颜色集合，可返回颜色
    #              fontdict={'weight': 'bold', 'size': 9})
    ax0 = ax.scatter(X[:,0], X[:,1], c=Y, marker='o', cmap='plasma')
    # ax0 = ax.scatter(X[:125,0], X[:125,1], c='r', marker='>', cmap='plasma')
    
    # ax0 = ax.scatter(X[:, 0], X[:, 1])
    fig.colorbar(ax0)

    for i in range(X.shape[0]):
        plt.annotate(str(Y[i]), xy=(X[i, 0], X[i, 1]))
 
    # plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)
    plt.savefig(save_path)
    print('done3')


if __name__ == '__main__':
    # plot_embedding(X_embedded[:2400, :], Y[:2400, :])
    # for i in range(10):
    X_item = X
    print(X_item)
    Y_item = Y
    print(Y_item)
    
    Save_item = os.path.join(rootpath, 'proto.jpg')
    
    # X_num = torch.load(X_item)
    # X_mean = torch.mean(X_num, dim=0)
    # X_num = (X_num-X_mean)/torch.norm((X_num-X_mean), dim=1, keepdim=True)
    # Y_num = torch.load(Y_item)
    
    X_num = np.array(X_item)
    # Y_num = np.array(Y_num)
    print("loaded")
    
    X_embed = TSNE(n_components=2, perplexity=5, init='pca', random_state=100).fit_transform(X_num)
    print("done1")
    plot_embedding(X_embed, Y_item, Save_item)
    print('done_all')
    
    

    









