import pickle
import os
import pandas as pd
import geopandas
import math
import numpy as np
import copy
import time
import seaborn as sns

import sklearn
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import hdbscan
from sklearn.mixture import GaussianMixture
import HDBSCAN_with_Direction_Centrality_Metric as HD
import sklearn.cluster as cluster
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import cluster_acc
import CDC
import shutil
def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # 删除文件或链接
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


sns.set_context('poster')
sns.set_color_codes()


def timeout_handler(signum, frame):
    print("算法已超时，正在终止...")
    raise Exception("算法超时")

def plot_clusters(data_name,data, algorithm, args, kwds):
    print(algorithm.__name__)

    labels_original=data[:,2]

    folder_path1 = r'/root/autodl-tmp/code2/cache/' + data_name
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)

    folder_path2 =r'/root/autodl-tmp/code2/result/'+algorithm.__name__
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)


    #超时跳出
    try:
        start_time = time.time()

        if algorithm.__name__!='HDCBC':
            if algorithm.__name__ == 'CDC':
                labels = algorithm(**kwds).fit_predict(data[:,:2])
            else:
                labels = algorithm(*args, **kwds).fit_predict(data[:,:2])

        else:

            labels = algorithm( **kwds).fit_predict(data_name,data)
            
        end_time = time.time()

        ars = adjusted_rand_score(labels_original, labels)
        nmi = normalized_mutual_info_score(labels_original, labels)
        acc = cluster_acc(labels_original, labels)
        time_use = end_time - start_time

        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plot_kwds = {'alpha': 1, 's': 5, 'linewidths': 0}

        fig, ax = plt.subplots()

        plt.scatter(data[:, 0], data[:, 1], c=colors, **plot_kwds)

        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

        #出图显示结果
        #plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)

        # text = 'ars:{:.2f}'.format(ars) + ',nmi:{:.2f}'.format(nmi) + ',acc:{:.2f}'.format(acc) + ',time:{:.2f}'.format(
        #     time_use)
        #
        # ax.text(0.05, 0.95, text, transform=ax.transAxes, color='black', fontsize=12)



    except Exception as e:
        print(e)
        ars='Null'
        nmi='Null'
        acc='Null'
        time_use='timeout'

        fig, ax = plt.subplots()
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
        text = f'ars:{ars}' + f',nmi:{ars}' + f',acc:{ars}' + f',time:{time_use}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, color='black', fontsize=12)


#对于所有挑选的数据集进行直接的计算
    if algorithm.__name__ == 'HDCBC':
        plt.savefig(folder_path2 + r'/' + data_name + '_' + algorithm.__name__ + '.png', dpi=1000)
        plt.close(fig)
        return 1,ars, nmi, acc,time_use

    else:
        plt.savefig(folder_path2 + r'/' + data_name + '_' + algorithm.__name__ + '.png', dpi=1000)
        plt.close(fig)

        return 1,ars, nmi, acc,time_use


#运行主程序，在这修改不同的参数，并进行不同数据集的对比（在这次的版本中，我们需要做的是将一个数据集进行一个横向的比较，以便能够达到一个相对合理的结果）
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.manifold import TSNE

    with open(r'/root/autodl-tmp/code2/cache/data_csv.pkl', 'rb') as file:
        data=pickle.load(file)

    results=[]
    attr = []

    for key in data.keys():
        acc1 = 0
        att ={}
        result={}
        list1 = [8]
        list2 = [5,6,7,8,9,10,12,15]
        list3 = [0.05,0.08,0.1,0.15,0.2]
        list4 = [5,6,7,8,9,10,12,15,18,20]

        data_name=key
        for i in list1:
            for j in  list2:
                for m in  list3:
                    for n1 in  list4:

                        parameter={'K_DCM': i,
                                   'K_nearest': j,
                                   'CM_threshold': m,
                                   'minclustersize': n1}

                        data_usage, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, HD.HDCBC, (), parameter)

                        if acc > acc1:
                            acc1 =acc
                            i1 =i
                            j1= j
                            m1 =m
                            n11=n1
                            
                    clear_folder(('./cache/' + data_name))

        att['k_DCM'] = i1
        att['K_nearest'] = j1
        att['CM_threshold'] = m1
        att['minclustersize'] = n11
        
        parameter={'K_DCM': i1,
                    'K_nearest': j1,
                    'CM_threshold': m1,
                    'minclustersize': n11}

        data_usage, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, HD.HDCBC, (), parameter)
        #对于不同的数据集，我们需要挑选出来相对正确的数据集的数量，从而达到一个
        result['HDCBC'] = [ars, nmi, acc, time_use]

        results.append([data_name, result])
        attr.append([data_name, att])

#后续为数据生成和保存的部分，无需修改
    ars_result = {}
    nmi_result = {}
    acc_result = {}
    time_use_result = {}

    ars_result['data_name'] = []
    nmi_result['data_name'] = []
    acc_result['data_name'] = []
    time_use_result['data_name'] = []

    for row in results:
        ars_result['data_name'].append(row[0])
        nmi_result['data_name'].append(row[0])
        acc_result['data_name'].append(row[0])
        time_use_result['data_name'].append(row[0])

        for key in row[1].keys():
            if key in ars_result:
                ars_result[key].append(row[1][key][0])
            else:
                ars_result[key] = [row[1][key][0]]

            if key in nmi_result:
                nmi_result[key].append(row[1][key][1])
            else:
                nmi_result[key] = [row[1][key][1]]

            if key in acc_result:
                acc_result[key].append(row[1][key][2])
            else:
                acc_result[key] = [row[1][key][2]]

            if key in time_use_result:
                time_use_result[key].append(row[1][key][3])
            else:
                time_use_result[key] = [row[1][key][3]]

    df_ars = pd.DataFrame(ars_result)
    df_nmi = pd.DataFrame(nmi_result)
    df_acc = pd.DataFrame(acc_result)
    df_attr = pd.DataFrame(attr)

    df_ars.to_csv('ars.csv')
    df_nmi.to_csv('nmi.csv')
    df_acc.to_csv('acc.csv')
    df_attr.to_csv('att_hdcbc.csv')
