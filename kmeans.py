from sklearn.cluster import KMeans
import numpy as np
from glob import glob
from data_load import get_mfccs_and_phones

data_path = "./TIMIT/*/*/*/*.WAV"
wav_files = glob(data_path)

mfcc_list = None
phns_list = None
count = 10

for wav_file in wav_files:
    if count == 0: break
    else: count -= 1

    if mfcc_list is None:
        mfcc_list, phns_list = get_mfccs_and_phones(wav_file)
    else:
        mfccs, phns = get_mfccs_and_phones(wav_file)
        mfcc_list = np.concatenate((mfcc_list, mfccs))
        phns_list = np.concatenate((phns_list, phns))
        
        
    
print(mfcc_list.shape)
print(phns_list.shape)


'''
X = np.array([[1, 2], [1, 4], [1, 0],
             [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)



print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)
'''
