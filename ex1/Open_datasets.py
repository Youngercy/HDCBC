
import numpy as np

import arff


import pandas

import os
import pickle




def noise_generate(data):

    original_data=data.values
    x_min, x_max = original_data[:, 0].min(), original_data[:, 0].max()
    y_min, y_max = original_data[:, 1].min(), original_data[:, 1].max()


    length=int(original_data.shape[0]/10)
    noise_x = np.random.uniform(x_min, x_max, length)
    noise_y = np.random.uniform(y_min, y_max, length)
    noise_label = np.full(length, -1)

    #print(original_data)
    nose=np.column_stack((noise_x, noise_y, noise_label))

    noisy_data = np.vstack((original_data,nose))
    pdf=pandas.DataFrame(noisy_data)

    return pdf



folder_path=r'.Experiment/ex1/datasets/'

file_list = os.listdir(folder_path)


data={}
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    if file_name.startswith('.'):
        continue
    file_name=file_name.replace('.csv','')

    print(file_path)

    if os.path.isfile(file_path):

        try:
            d=pandas.read_csv(file_path)
            #print(d.values[:,0].min())

            d=noise_generate(d)


            data[file_name]=d
            print(d.head())
        except ZeroDivisionError as e:

            print("Division by zero:", e)
        except ValueError as e:

            print("Value error:", e)
        except Exception as e:

            print("An error occurred:", e)

data_list={}
for key in data.keys():

    data_list[key] = data[key]




print(len(data_list))

with open(r'./Experiment/ex1/cache/data_csv.pkl','wb') as file:
    pickle.dump(data_list,file)






