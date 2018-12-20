# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat

import json
from metric_learn import mmc, mlkr
import numpy as np
from pr_function import knn_classifier, kmeans_classifier, compute_eigenspace

#-------------------------------------------------------------------------------------
# Load images data
camId = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['camId'].flatten())
filelist = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten())
labels = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['labels'].flatten())
gallery_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten())
query_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten())
train_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten())
data_size = camId.shape[0]
class_size = 700

# Load features data
with open('feature_data.json', 'r') as f:
    features = np.array(json.load(f))

# Data preparation
train_idx = train_idx - 1
query_idx = query_idx - 1
gallery_idx = gallery_idx - 1

train_size = train_idx.shape[0]
query_size = query_idx.shape[0]
gallery_size = gallery_idx.shape[0]

train_cam = camId[train_idx]
query_cam = camId[query_idx]
gallery_cam = camId[gallery_idx]

train_label = labels[train_idx]
query_label = labels[query_idx]
gallery_label = labels[gallery_idx]

train_data = features[train_idx,:]
query_data = features[query_idx,:]
gallery_data = features[gallery_idx,:]

#-------------------------------------------------------------------------------------
M = 35
rank = 1

#base line
query_trans = query_data
gallery_trans = gallery_data
test_knn_score, test_mAP_score = knn_classifier(rank, query_size, query_trans, query_label, query_cam, gallery_size, gallery_trans, gallery_cam, gallery_label)
print('Baseline Result, knn_success_rate = ', test_knn_score, 'where k = ', rank)
print('Baseline Result, knn_mAP = ', test_mAP_score, 'where k = ', rank)
test_kmeans_score = kmeans_classifier(class_size, query_size, query_trans, query_label, gallery_size, gallery_trans, gallery_label )
print('Baseline Result, kmeans score = ', test_kmeans_score)

# PCA
print("PCA started...")
A_train, e_vals, e_vecs = compute_eigenspace(train_data, "high")
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T
m_vecs = e_vecs[:,0:M]
train_pca = np.dot(train_data, m_vecs)
query_pca = np.dot(query_data, m_vecs)
gallery_pca = np.dot(gallery_data, m_vecs)
test_knn_score, test_mAP_score = knn_classifier(rank, query_size, query_pca, query_label, query_cam, gallery_size, gallery_pca, gallery_cam, gallery_label)
print('PCA Result, knn_success_rate = ', test_knn_score, 'where k = ', rank)
print('PCA Result, knn_mAP = ', test_mAP_score, 'where k = ', rank)
test_kmeans_score = kmeans_classifier(class_size, query_size, query_pca, query_label, gallery_size, gallery_pca, gallery_label )
print('PCA Result, kmeans score = ', test_kmeans_score)

# Kernel
print("Kernel started...")
kern = mlkr.MLKR(max_iter = 1)
kern.fit(train_pca, train_label)
query_trans = kern.transform(query_pca)
gallery_trans = kern.transform(gallery_pca)
test_knn_score, test_mAP_score = knn_classifier(rank, query_size, query_trans, query_label, query_cam, gallery_size, gallery_trans, gallery_cam, gallery_label)
print('Kernel Result, knn_success_rate = ', test_knn_score, 'where k = ', rank)
print('Kernel Result, knn_mAP = ', test_mAP_score, 'where k = ', rank)
test_kmeans_score = kmeans_classifier(class_size, query_size, query_trans, query_label, gallery_size, gallery_trans, gallery_label )
print('Kernel Result, kmeans score = ', test_kmeans_score)

# Mahalonobis
print("Mahalonobis started...")
maha_sup = mmc.MMC_Supervised(max_iter = 3, num_constraints = 1)
maha_sup.fit(train_pca, train_label)
A = maha_sup.metric()
query_trans = maha_sup.transform(query_pca)
gallery_trans = maha_sup.transform(gallery_pca)
test_knn_score, test_mAP_score = knn_classifier(rank, query_size, query_trans, query_label, query_cam, gallery_size, gallery_trans, gallery_cam, gallery_label)
print('Mahalonobis Result, knn_success_rate = ', test_knn_score, 'where k = ', rank)
print('Mahalonobis Result, knn_mAP = ', test_mAP_score, 'where k = ', rank)
test_kmeans_score = kmeans_classifier(class_size, query_size, query_trans, query_label, gallery_size, gallery_trans, gallery_label )
print('Mahalonobis Result, kmeans score = ', test_kmeans_score)

#-------------------------------------------------------------------------------------
label_trunc = np.unique(train_label)
val_num = 7
print("Validation started...")
itr = [1, 10]
val_knn_score = np.zeros((len(itr), val_num))
val_mAP_score = np.zeros((len(itr), val_num))
val_kmean_score = np.zeros((len(itr), val_num))
for v in range(val_num):
    # Non replace selection of validation set
    val_test_class= np.random.choice(label_trunc, 100, replace = False)
    trunc_delete_idx = [ np.argwhere(label_trunc == t) for t in val_test_class ]
    label_trunc = np.delete(label_trunc, trunc_delete_idx)
    # Extract data and label for each subset
    train_delete_idx =  [ i for t in val_test_class for i in range(train_size) if train_label[i] == t ]
    val_train_label = np.delete(train_label, train_delete_idx)
    val_train_data = np.delete(train_data, train_delete_idx, axis = 0)
    val_test_label = train_label[train_delete_idx]
    val_test_cam = train_cam[train_delete_idx]
    val_test_data = train_data[train_delete_idx,:]
    # Spit query and gallery sets
    query_rand_idx = [ i for t in val_test_class for i in np.random.choice( np.argwhere(val_test_label == t).flatten(), 2, replace = False ) ]
    val_query_label = val_test_label[query_rand_idx]
    val_query_data = val_test_data[query_rand_idx,:]
    val_query_size = val_query_label.shape[0]
    val_query_cam = val_test_cam[query_rand_idx]
    
    val_gallery_label = np.delete(val_test_label, query_rand_idx)
    val_gallery_data = np.delete(val_test_data, query_rand_idx, axis = 0)
    val_gallery_size = val_gallery_label.shape[0]
    val_gallery_cam = np.delete(val_test_cam, query_rand_idx)
    
    # PCA
    A_train, e_vals, e_vecs = compute_eigenspace(val_train_data, "high")
    idx=np.argsort(np.absolute(e_vals))[::-1]
    e_vals = e_vals[idx]
    e_vecs = (e_vecs.T[idx]).T
    m_vecs = e_vecs[:,0:M]
    train_pca = np.dot(val_train_data, m_vecs)
    query_pca = np.dot(val_query_data, m_vecs)
    gallery_pca = np.dot(val_gallery_data, m_vecs)
    
    for r in range(len(itr)):
        '''
        # Kernel
        kern = mlkr.MLKR(max_iter = itr[r])
        kern.fit(train_pca, val_train_label)
        query_trans = kern.transform(query_pca)
        gallery_trans = kern.transform(gallery_pca)
        '''
        # Mahalonobis
        maha_sup = mmc.MMC_Supervised(max_iter = itr[r], num_constraints = 1)
        maha_sup.fit(train_pca, val_train_label)
        A = maha_sup.metric()
        query_trans = maha_sup.transform(query_pca)
        gallery_trans = maha_sup.transform(gallery_pca)
        
        # Classification knn
        knn_score, mAP_score = knn_classifier(rank, val_query_size, query_trans, val_query_label, val_query_cam, val_gallery_size, gallery_trans, val_gallery_cam, val_gallery_label)
        val_knn_score[r,v] = knn_score
        val_mAP_score[r,v] = mAP_score
        # Classification knn
        #kmeans_score = kmeans_classifier(100, val_query_size, query_trans, val_query_label, val_gallery_size, gallery_trans, val_gallery_label )
        #val_kmean_score[r,v] = kmeans_score
    print("Validation ", v, " finished!")

#Choose the optimal parameter with knn_success_rate
val_knn_score = np.mean(val_knn_score, axis = 1)
opt_itr = itr[np.argmax(val_knn_score)]

#Choose the optimal parameter with mAP
#val_mAP_score = np.mean(val_mAP_score, axis = 1)
#opt_itr = itr[np.argmax(val_mAP_score)]

#Choose the optimal parameter with k-means
#val_kmean_score = np.mean(val_kmean_score, axis = 1)
#opt_itr = itr[np.argmax(val_kmean_score)]

# PCA
A_train, e_vals, e_vecs = compute_eigenspace(train_data, "high")
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T
m_vecs = e_vecs[:,0:M]
train_pca = np.dot(train_data, m_vecs)
query_pca = np.dot(query_data, m_vecs)
gallery_pca = np.dot(gallery_data, m_vecs)


# Mahalonobis
print("Final Mahalonobis Starting")
maha_sup = mmc.MMC_Supervised(max_iter = opt_itr, num_constraints = 1)
maha_sup.fit(train_pca, train_label)
A = maha_sup.metric()
query_trans = maha_sup.transform(query_pca)
gallery_trans = maha_sup.transform(gallery_pca)
'''
# Kernel
print("Final Kernel started...")
kern = mlkr.MLKR(max_iter = opt_itr)
kern.fit(train_pca, train_label)
query_trans = kern.transform(query_pca)
gallery_trans = kern.transform(gallery_pca)
'''
# Classification
print("Starting classification")
test_knn_score, test_mAP_score = knn_classifier(rank, query_size, query_trans, query_label, query_cam, gallery_size, gallery_trans, gallery_cam, gallery_label)
print('Final Result, knn_success_rate = ', test_knn_score, 'where k = ', rank)
print('Final Result, knn_mAP = ', test_mAP_score, 'where k = ', rank)
test_kmeans_score = kmeans_classifier(class_size, query_size, query_trans, query_label, gallery_size, gallery_trans, gallery_label )
print('Final Result, kmeans score = ', test_kmeans_score)