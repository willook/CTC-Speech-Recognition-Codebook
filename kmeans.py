from sklearn.cluster import KMeans
import numpy as np
from glob import glob
from data_load import get_mfccs_and_phones
import matplotlib.pyplot as plt
count = 1
n_dim = 2
visualize = False
data_path = "./TIMIT/*/*/*/*.WAV"
wav_files = glob(data_path)

mfcc_list = None
phns_list = None


for wav_file in wav_files:
    if count == 0: break
    else: count -= 1

    if mfcc_list is None:
        mfcc_list, phns_list = get_mfccs_and_phones(wav_file)
    else:
        mfccs, phns = get_mfccs_and_phones(wav_file)
        mfcc_list = np.concatenate((mfcc_list, mfccs))
        phns_list = np.concatenate((phns_list, phns))
        
mfcc_list = mfcc_list[:,1:1+n_dim]
model = KMeans(n_clusters=8, init='k-means++',max_iter=300)
kmeans = model.fit(mfcc_list)
predicts = kmeans.predict(mfcc_list)

if visualize:
    centers = kmeans.cluster_centers_
    center_x = centers[:,0]
    center_y = centers[:,1]
    
    plt.scatter(mfcc_list[:,0],mfcc_list[:,1],c = predicts, alpha = 0.5)
    plt.scatter(center_x,center_y,s=50, marker='D', c='r')
    plt.savefig("./kmeans distribution with k=8.png")

    plt.scatter(mfcc_list[:,0],mfcc_list[:,1],c = phns_list, alpha = 0.5)
    plt.scatter(center_x,center_y,s=50, marker='D', c='r')
    plt.savefig("./phonemes distribution with k=8.png")

print(mfcc_list.shape)
print(phns_list.shape)
print(kmeans.cluster_centers_)
print(mfcc_list[:,0].shape,mfcc_list[:,1].shape,predicts.shape)

'''
X = np.array([[1, 2], [1, 4], [1, 0],
             [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)



print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)
'''
