import scipy.io as sio
import math
import numpy as np
import random
dir = "~/bearing_data_set/"
data_path = {
    'Inner_Race_0.007':"105.mat",
    'Inner_Race_0.014':"169.mat",
    'Inner_Race_0.021':"209.mat",
    'Inner_Race_0.028':"3001.mat",
    'Ball_0.007': "118.mat",
    'Ball_0.014': "185.mat",
    'Ball_0.021': "222.mat",
    'Ball_0.028': "3005.mat",
    'Outer_Race_0.007':"156.mat",
    'Outer_Race_0.021': "258.mat"
}


class Bearing_dataset:
    depart = 4/5
    
    def __init__(self):
        self.labels = {0:'Inner_Race_0.007',
                       1:'Inner_Race_0.014',
                       2:'Inner_Race_0.021',
                       3:'Inner_Race_0.028',
                       4:'Ball_0.007',
                       5:'Ball_0.014',
                       6:'Ball_0.021',
                       7:'Ball_0.028',
                       8:'Outer_Race_0.007',
                       9:'Outer_Race_0.021'}
        
        self.data_with_labels = self._match_label_data()
        self.data_label_lst = self._construct_list()
        random.shuffle(self.data_label_lst)
        self.data_lst, self.label_lst = self._depart_data_label()
        self.data_matrix = self._construct_data_matrix()
        self.matrix_normalization()
        self.train_data = self.data_matrix[: 7000, :]
        self.eval_data = self.data_matrix[7000 : , :]
        self.train_label = self.label_lst[: 7000]
        self.eval_label = self.label_lst[7000 : ]
    def _match_label_data(self):
        final = 1000
        step = 1200 
        dic = {}
        for n, s in list(self.labels.items()):
            dic[n] = sio.loadmat(data_path.get(s))
            for key in list(dic[n].keys()):
                if key.endswith('DE_time'):
                    dic[n] = dic[n].get(key)
                    dic[n] = np.transpose(self._truncate(dic[n]))
                    sub_lst = []
                    for i in range(final):
                        scale = random.randint(1, 100)
                        t = i * scale
                        sub_lst.append(dic[n][:,t:t+step])
                    dic[n] = sub_lst
                    random.shuffle(dic[n])
                   # dic[n] = np.split(dic[n], 100, axis = 1)
                   # for elem in dic[n]:
                    #    elem = np.transpose(elem)
        return dic
    def _truncate(self,array):
        return array[:120000, :]

    def _construct_list(self):
        lst = []
        for label in self.labels.keys():
            for elem in self.data_with_labels[label]:
                lst.append((label, elem))
        return lst
    def _depart_data_label(self):
        data_lst  = []
        label_lst = []
        for elem in self.data_label_lst:
            data_lst.append(elem[1])
            label_lst.append(elem[0])
        return data_lst, label_lst
    def _construct_data_matrix(self):
        elem = self.data_lst[0]
        for i in range(1, len(self.data_lst)):
            elem = np.concatenate((elem, self.data_lst[i]), axis = 0)
        return elem

    def matrix_normalization(self):
        data_mean = self.data_matrix.mean(axis = 0)
        self.data_matrix -= data_mean
        self.data_matrix /= np.std(self.data_matrix, axis = 0)

#  def _construct_list(self):
#      lst = []
#      for label in self.labels.keys():
#          for i in range(self.data_with_labels[label].shape[0]):
#              lst.append([label, self.data_with_labels[label][i,0]])
#      return lst

#
#
#



Bearing_dataset = Bearing_dataset()
