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
import HDCBC as HD
import sklearn.cluster as cluster
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import cluster_acc
import CDC

sns.set_context('poster')
sns.set_color_codes()

def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # 删除文件或链接
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def timeout_handler(signum, frame):
    print("time out")
    raise Exception("time out")

def plot_clusters(data_name,data, algorithm, args, kwds):
    print(algorithm.__name__)

    labels_original=data[:,2]

    folder_path1 = r'./Experiment/ex2/noise_free/cache/' + data_name
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)

    folder_path2 = r'./Experiment/ex2/noise_free/result/'+algorithm.__name__
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


#Comparison of different Algorithms
    if algorithm.__name__ == 'HDCBC':
        plt.savefig(folder_path2 + r'/' + data_name + '_' + algorithm.__name__ + '.png', dpi=1000)
        plt.close(fig)
        return 1,ars, nmi, acc,time_use

    else:
        plt.savefig(folder_path2 + r'/' + data_name + '_' + algorithm.__name__ + '.png', dpi=1000)
        plt.close(fig)

        return 1,ars, nmi, acc,time_use


#run main
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.manifold import TSNE

    #the original data
    with open(r'./Experiment/ex2/noise_free/cache/data_csv.pkl', 'rb') as file:
        data=pickle.load(file)


    results=[]
    attr = []

#test for HDCBC
    for key in data.keys():
        acc1 = 0
        att = {}
        result = {}
        list1 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        list2 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        list3 = [0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1,0.15,0.2]
        list4 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,50]

        data_name = key
        for i in list1:
            for j in list2:
                for m in list3:
                    for n1 in list4:

                        parameter = {'K_DCM': i,
                                     'K_nearest': j,
                                     'CM_threshold': m,
                                     'minclustersize': n1}

                        data_usage, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, HD.HDCBC, (),
                                                                            parameter)

                        if acc > acc1:
                            acc1 = acc
                            i1 = i
                            j1 = j
                            m1 = m
                            n11 = n1

                    clear_folder(('./cache/' + data_name))

        att['k_DCM'] = i1
        att['K_nearest'] = j1
        att['CM_threshold'] = m1
        att['minclustersize'] = n11

        parameter = {'K_DCM': i1,
                     'K_nearest': j1,
                     'CM_threshold': m1,
                     'minclustersize': n11}

        data_usage, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, HD.HDCBC, (), parameter)
        result['HDCBC'] = [ars, nmi, acc, time_use]

        results.append([data_name, result])
        attr.append([data_name, att])



    #set the parameters for Kmeans and Agglomerative clustering
        num_t = 2

        if data_name == 'banana':
            num_t = 2
        if data_name == 't4.8k':
            num_t = 6
        if data_name == 't8.8k':
            num_t = 8
        if data_name == 'pearl':
            num_t = 3
        if data_name == 'donut3':
            num_t = 3
        if data_name == 'diamond9':
            num_t = 9
        if data_name == 'twenty':
            num_t = 20
        if data_name == 'target':
            num_t = 2
        if data_name == 't5.8k':
            num_t = 6
        if data_name == 'curves1':
            num_t = 2
        if data_name == 'twodiamonds':
            num_t = 2
        if data_name == 'curves2':
            num_t = 2
        if data_name == 'zelnik3':
            num_t = 3
        if data_name == 'donutcurves':
            num_t = 4
        if data_name == 'spherical_6_2':
            num_t = 6

        _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, cluster.KMeans, (),
                                                   {'n_clusters': num_t})
        _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, cluster.AgglomerativeClustering, (),
                                                   {'n_clusters': num_t, 'linkage': 'ward'})
        result['AgglomerativeClustering'] = [ars, nmi, acc, time_use]

    #test for DBSCAN
        acc0 = 0
        list1 = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]
        list0 = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
        for eps in list1:
            for minsam in list0:

                _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, cluster.DBSCAN, (),
                                                           {'eps': eps, 'min_samples': minsam})
                if acc > acc0:
                    acc0 = acc
                    eps1 = eps
                    minsam1 = minsam

        _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, cluster.DBSCAN, (),
                                                   {'eps': eps1, 'min_samples': minsam1})
        result['DBSCAN'] = [ars, nmi, acc, time_use]
        att1['eps'] = eps1
        att1['min_samples'] = minsam1

    ## test for MeanShift
        acc1 = 0
        list2 = [0.1, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for bandwidth in list2:
            _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, cluster.MeanShift, (),
                                                       {'bandwidth': bandwidth, 'cluster_all': False})

            if acc > acc1:
                bandwidth1 = bandwidth
                acc1 = acc
        _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, cluster.MeanShift, (),
                                                   {'bandwidth': bandwidth1, 'cluster_all': False})

        result['MeanShift'] = [ars, nmi, acc, time_use]
        att1['band'] = bandwidth1


    #test of CDC
        acc3 = 0
        list1 = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 100]
        list0 = [0.01, 0.02, 0.03,0.04, 0.05,0.06,0.07, 0.08,0.09, 0.1, 0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19, 0.2]
        for k_num in list1:
            for t_dcm in list0:

                _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, CDC.CDC, (),
                                                           {'k_num': k_num, 'T_DCM': t_dcm})
                if acc > acc3:
                    acc3 = acc
                    k_num1 = k_num
                    t_dcm1 = t_dcm
        _, ars, nmi, acc, time_use = plot_clusters(data_name, data[key].values, CDC.CDC, (),
                                                   {'k_num': k_num1, 'T_DCM': t_dcm1})
        result['CDC'] = [ars, nmi, acc, time_use]

        results.append([data_name, result])
        attr.append([data_name, att1])

        # generation results
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
    df_time_use = pd.DataFrame(time_use_result)
    att_out = pd.DataFrame(attr)
    att_out.to_csv('att.csv')
    df_ars.to_csv('ars.csv')
    df_nmi.to_csv('nmi.csv')
    df_acc.to_csv('acc.csv')
    df_time_use.to_csv('time_use.csv')