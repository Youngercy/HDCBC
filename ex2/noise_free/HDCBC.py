import copy

import geopandas

import pandas as pd

import sklearn.metrics

from scipy.spatial import cKDTree
import os
import pickle
import hdbscan
import math

import matplotlib.pyplot as plt

import numpy as np

from sklearn.mixture import GaussianMixture



pd.set_option('display.max_columns', None)

class HDCBC:

    def __init__(self,K_DCM,K_nearest,CM_threshold,minclustersize):

        self.K_DCM=K_DCM
        self.K_nearest=K_nearest
        self.CM_threshold=CM_threshold
        self.minclustersize=minclustersize
        self.minimum_noise_size=3
        self.k_noise_density = 3

    def fit_predict(self,data_name,data:pd.DataFrame):

        coordinates = self.array_to_coordinates(data, data_name)
        DCM_core_distance_list = self.kth_nearest_DCM_point(data_name=data_name, coordinates=coordinates, K_DCM=self.K_DCM,
                                                       K_nearest=self.K_nearest, DCM_threshold=self.CM_threshold)

=
        matirx = self.compute_reachability_distance_matrix(data_name=data_name, coordinates=coordinates \
                                                      , DCM_core_distance_list=DCM_core_distance_list)

        noise_indices, cluster_indices, labels_raw = self.HDBSCAN_matrix_only(data_name=data_name,coordinates=coordinates, distance_matrix=matirx, \
                                                                        if_draw_graph=False, minclustersize=self.minclustersize)

        labels=self.map_labels_to_integers(labels_raw)

        labels_redistribute = self.noise_to_nearest_cluster(noise_indices=noise_indices, cluster_indices=cluster_indices,
                                                       labels=labels, coordinates=coordinates)
        noise_dic = self.noise_density_get(coordinates=coordinates, tree_path=r'./cache/' + data_name + r'/ckDtree.pkl' \
                                      , labels_redistribute=labels_redistribute, \
                                      noise_indices=noise_indices, minimum_noise_size=self.minimum_noise_size, k_noise_density=self.k_noise_density)
        labels_recluster = self.noise_recluster(labels=labels, noise_dic=noise_dic)

        '''
        self.draw_graph(data_name=self.data_name, append='_new', labels=labels_recluster, coordinates=self.coordinates, if_label=True,
                   if_save=True, if_show=False)
        '''

        return labels_recluster


    def poi_to_coordinate(self,data_name, data_path=r'./data/poi_futian_part.shp'):
        gdf_poi = geopandas.read_file(data_path)
        target_crs = 'EPSG:4547'

        gdf_poi = gdf_poi.to_crs(target_crs)

        if not os.path.exists(r'./cache/' + data_name + r'/coordinates.pkl'):
            # 提取gdf的坐标
            coordinates = gdf_poi.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
            with open(r'./cache/' + data_name + r'/coordinates.pkl', 'wb') as file2:
                pickle.dump(coordinates, file2)
        else:
            with open(r'./cache/' + data_name + r'/coordinates.pkl', 'rb') as file2:
                coordinates = pickle.load(file2)

        return gdf_poi, coordinates


    def array_to_coordinates(self,data, data_name):

        coordinates = []
        for row in data:
            coordinates.append((row[0], row[1]))
        with open(r'./cache/' + data_name + r'/coordinates.pkl', 'wb') as file2:
            pickle.dump(coordinates, file2)

        return coordinates

    def poi_cKDTree_build(self,coordinates, data_name):

        if not os.path.exists(r'./cache/' + data_name + r'/ckDtree.pkl'):
            print('building_ckDtree')

            tree = cKDTree(coordinates)
            with open(r'./cache/' + data_name + r'/ckDtree.pkl', 'wb') as file:
                pickle.dump(tree, file)
        else:
            with open(r'./cache/' + data_name + r'/ckDtree.pkl', 'rb') as file:
                tree = pickle.load(file)

        return tree


    def find_KNN_points(self,data_name, coordinates, k):

        tree = self.poi_cKDTree_build(coordinates, data_name=data_name)
        coordinates_with_KNN_points = []
        for coor in coordinates:
            points = []
            _, indices = tree.query(coor, k=k + 1)
            for indice in indices:
                points.append(coordinates[indice])
            coordinates_with_KNN_points.append(points)

        return coordinates_with_KNN_points

    def calculate_distance_spatial(self,point1, point2):
        x1, y1 = point1

        x2, y2 = point2

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def match_DCM(self,DCM1, DCM2, threshold):
        if np.abs(DCM1 - DCM2) < threshold:
            return True
        else:
            return False

    # k+1 point
    def calculate_angles(self,points):

        center_point = points[0]
        angles = []
        angles_right = []
        for i in range(1, len(points)):
            vector = [points[i][0] - center_point[0], points[i][1] - center_point[1]]

            angle = np.arctan2(vector[1], vector[0])
            if angle < 0:
                angle = angle + np.pi * 2
            angles.append(angle)
        angles = sorted(angles)

        for j in range(len(angles)):
            if j == len(angles) - 1:
                a = np.pi * 2.0 - (angles[j] - angles[0])
            else:
                a = angles[j + 1] - angles[j]
            angles_right.append(a)

        return angles_right

    def local_direction_centrality(self,angle_list):

        k = len(angle_list)
        nkp = 2.0 * np.pi / (k)
        result = 0
        for angle in angle_list:
            result += np.square(angle - nkp)

        result = result * k / (4 * (k - 1) * np.square(np.pi))

        return result

    def kth_nearest_DCM_point(self,data_name, coordinates, K_DCM, K_nearest, DCM_threshold):

        if not os.path.exists(r'./cache/' + data_name + r'/DCM_core_distance_list.pkl'):

            print('calculating DCM')
            tree = self.poi_cKDTree_build(coordinates=coordinates, data_name=data_name)

            coordinates_with_KNN_points = self.find_KNN_points(data_name=data_name, coordinates=coordinates, k=K_DCM)

            DCM_list = []
            core_distances = []
            for ckp in coordinates_with_KNN_points:
                c = self.calculate_angles(ckp)
                c = self.local_direction_centrality(c)
                DCM_list.append(c)


            print('calculating core distance')
            l = len(coordinates)
            for i in range(l):
                # print(i)
                count = 0
                k_count = 0
                core_distance = -1

                while k_count < K_nearest:

                    count += 1

                    distance_list, indices = tree.query(coordinates[i], k=count + 1)

                    core_distance = distance_list[-1]

                    if self.match_DCM(DCM_list[i], DCM_list[indices[-1]], threshold=DCM_threshold):
                        k_count += 1
                        # print(k_count)

                    if count >= l - 20:
                        break
                if k_count == K_nearest:
                    core_distances.append(core_distance)

                else:
                    core_distances.append(9999999999999)

            with open(r'./cache/' + data_name + r'/DCM_core_distance_list.pkl', 'wb') as file:
                pickle.dump(core_distances, file)
        else:
            print('calculating core distance')
            with open(r'./cache/' + data_name + r'/DCM_core_distance_list.pkl', 'rb') as file:
                core_distances = pickle.load(file)

        return core_distances

    def compute_reachability_distance_matrix(self,data_name, coordinates, DCM_core_distance_list):
        print('building reachability matrix')
        if not os.path.exists(r'./cache/' + data_name + r'/reachability_distance_matrix.pkl'):

            num_points = len(coordinates)


            distance_matrix = sklearn.metrics.pairwise_distances(coordinates)
            core_matrix_row = np.tile(DCM_core_distance_list, (num_points, 1))
            core_matrix_column = core_matrix_row.transpose()

            distance_matrix = np.maximum(distance_matrix, core_matrix_row, core_matrix_column)

            '''
            distance_matrix=np.zeros((num_points,num_points))
            for i in range(num_points):
                #print(i)

                for j in range(i+1, num_points):
                    spatial_distance = calculate_distance_spatial(coordinates[i], coordinates[j])

                    core_distance_i = DCM_core_distance_list[i]
                    core_distance_j = DCM_core_distance_list[j]
                    if core_distance_i==-1 or core_distance_j==-1:
                        core_distance_i=9999999999999999
                    distance=max(spatial_distance,core_distance_i,core_distance_j)

                    distance_matrix[i][j] = distance

                    distance_matrix[j][i] = distance

            '''

            with open(r'./cache/' + data_name + r'/reachability_distance_matrix.pkl', 'wb') as file3:
                pickle.dump(distance_matrix, file3)
        else:
            with open(r'./cache/' + data_name + r'/reachability_distance_matrix.pkl', 'rb') as file3:
                distance_matrix = pickle.load(file3)

        return distance_matrix

    def draw_graph(self,data_name, append, labels, coordinates, if_label=False, if_save=True, if_show=False):

        noise_indices = np.atleast_1d(labels == -1).nonzero()[0]

        cluster_indices = np.atleast_1d(labels != -1).nonzero()[0]

        data1 = []
        data2 = []
        data = []
        for d in coordinates:
            dd = [d[0], d[1]]
            data.append(dd)
            data1.append(d[0])
            data2.append(d[1])


        fig = plt.figure()


        cmap = plt.cm.get_cmap('tab10', len(set(labels)))
        vmin = min(labels)
        vmax = max(labels)



        gray_color = (0.0, 0.0, 0.0)

        data1_notnoise = [data1[i] for i in cluster_indices]
        data2_notnoise = [data2[i] for i in cluster_indices]

        cluster_labels = [labels[i] for i in cluster_indices]

        if len(noise_indices) > 0:
            data1_noise = [data1[i] for i in noise_indices]
            data2_noise = [data2[i] for i in noise_indices]
            plt.scatter(data1_noise, data2_noise, color=gray_color, s=0.3)

        plt.scatter(data1_notnoise, data2_notnoise, c=cluster_labels, cmap=cmap, s=0.3, vmin=vmin, vmax=vmax)

        if if_label:
            for i, label in enumerate(labels):
                plt.annotate(label, (data[i][0], data[i][1]), textcoords="offset points", xytext=(0, 1), ha='center',
                             fontsize=1)

        if if_save:
            plt.savefig(r'./result/HDCBC/' + data_name + r'/' + data_name + append + '.png', dpi=1000)

        if if_show:
            plt.show()

        plt.close(fig)

    def HDBSCAN_matrix_only(self,data_name,coordinates, distance_matrix, if_draw_graph, minclustersize):
        print('performing hdbscan')
        if os.path.exists(r'./cache/' + data_name + r'/clusterer.pkl'):
            os.remove(r'./cache/' + data_name + r'/clusterer.pkl')
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=minclustersize, gen_min_span_tree=True)
        clusterer.fit(distance_matrix)
        with open(r'./cache/' + data_name + r'/clusterer.pkl', 'wb') as file:
            pickle.dump(clusterer, file)


        labels = clusterer.labels_

        noise_indices = np.where(labels == -1)[0]

        cluster_indices = np.where(labels != -1)[0]

        if if_draw_graph:
            self.draw_graph(data_name=data_name, labels=labels, coordinates=coordinates)

        return noise_indices, cluster_indices, labels

    def noise_to_nearest_cluster(self,noise_indices, cluster_indices, labels, coordinates):

        cluster_points = []
        cluster_labels = []

        labels_redistribute = copy.deepcopy(labels)

        for indice in cluster_indices:
            cluster_points.append(coordinates[indice])
            cluster_labels.append(labels[indice])

        Tree = cKDTree(cluster_points)

        for indice in noise_indices:
            _, nearest_indice = Tree.query(coordinates[indice], k=1)
            labels_redistribute[indice] = cluster_labels[nearest_indice]

        return labels_redistribute

    def noise_density_get(self,coordinates, tree_path, labels_redistribute, noise_indices, minimum_noise_size, k_noise_density):


        noise_indice_density_dic = {}

        all_cluster_dic = {}

        for indice in noise_indices:
            if labels_redistribute[indice] in noise_indice_density_dic.keys():
                noise_indice_density_dic[labels_redistribute[indice]].append(indice)
            else:
                noise_indice_density_dic[labels_redistribute[indice]] = [indice]

        for indice in range(len(labels_redistribute)):
            if labels_redistribute[indice] in all_cluster_dic.keys():
                all_cluster_dic[labels_redistribute[indice]].append(indice)
            else:
                all_cluster_dic[labels_redistribute[indice]] = [indice]

        keys = list(noise_indice_density_dic.keys())

        for key in keys:

            if len(noise_indice_density_dic[key]) < minimum_noise_size:
                value = noise_indice_density_dic.pop(key)
                if -1 in noise_indice_density_dic.keys():
                    noise_indice_density_dic[-1] = noise_indice_density_dic[-1] + value
                else:
                    noise_indice_density_dic[-1] = value
                continue

            cluster_i_points = []
            for indice in all_cluster_dic[key]:
                cluster_i_points.append(coordinates[indice])

            with open(tree_path, 'rb') as file2:
                tree = pickle.load(file2)

            for index, noise_indice in enumerate(noise_indice_density_dic[key]):
                distances, _ = tree.query(coordinates[noise_indice], k_noise_density + 1)
                noise_indice_density_dic[key][index] = [noise_indice, 1.0 / distances[-1]]

        return noise_indice_density_dic

    def Gaussian_density_based_bisection(self,noise_indice_density_list):
        os.environ["OMP_NUM_THREADS"] = "1"

        noise_not_any_more = []

        data = np.array(noise_indice_density_list)

        gmm = GaussianMixture(n_components=2, init_params='random')
        gmm.fit(data[:, 1].reshape(-1, 1))
        labels_0_1 = gmm.predict(data[:, 1].reshape(-1, 1))

        mean_density = {}
        for l in set(labels_0_1):
            mean_density[l] = np.mean(data[labels_0_1 == l, 1])

        max_density_label = max(mean_density, key=mean_density.get)

        for index, value in enumerate(labels_0_1):
            if value == max_density_label:
                noise_not_any_more.append(noise_indice_density_list[index])

        return noise_not_any_more

    def noise_recluster(self,labels, noise_dic):
        labels_recluster = copy.deepcopy(labels)

        for key in noise_dic.keys():
            if int(key) == -1:
                continue

            for point in self.Gaussian_density_based_bisection(noise_dic[key]):
                labels_recluster[int(point[0])] = int(key)

        return labels_recluster

        # coordinates = array_to_coordinates(d['data'],data_name)
        # label_real=[row[2] for row in d['data']]

        #coordinates = hdcm.array_to_coordinates(data[key].values, data_name)
        '''
        label_real=[row[2] for row in data[key].values]

        fig1=plt.figure()
        vmin = min(label_real)
        vmax = max(label_real)
        cmap = plt.cm.get_cmap('tab10', len(set(label_real)))
        # 设置灰色的RGB值
        # 绘制散点图，并根据聚类标签进行着色

        x=[point[0] for point in coordinates]
        y = [point[1] for point in coordinates]
        plt.scatter(x=x,y=y, c=label_real, cmap=cmap, s=0.3,vmin=vmin,vmax=vmax)
        plt.savefig(r'./cache/' + data_name + r'/' + data_name + '_original' + '.png', dpi=300)
        plt.close(fig1)

        '''
    def map_labels_to_integers(self,labels):

        labels_integer=[]
        unique_labels = sorted(list(set(labels)))
        for label in labels:
            if label==-1:
                labels_integer.append(-1)
                continue
            else:
                for i,value in enumerate(unique_labels):
                    if label==value:
                        labels_integer.append(i)
                        break

        labels_integer=np.array(labels_integer)

        return labels_integer

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.manifold import TSNE

    with open(r'D:\my_program\HDBSCAN_RENEW\cache\data_csv.pkl', 'rb') as file:
        data=pickle.load(file)

    for key in data.keys():

        data_name=key
        folder_path1=r'D:/my_program/HDBSCAN_RENEW/cache/'+data_name
        if not os.path.exists(folder_path1):
            os.mkdir(folder_path1)



        folder_path2=r'D:/my_program/HDBSCAN_RENEW/result/HDCBC/'+data_name
        if not os.path.exists(folder_path2):
            os.makedirs(folder_path2)

        hdcb=HDCBC(7,5,0.2,15)
        hdcb.fit_predict(data_name,data[key].values)







