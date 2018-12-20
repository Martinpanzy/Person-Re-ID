#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:58:26 2018

@author: panzengyang
"""

from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_eigenspace(X_data, mode):
    # Using high/low dimensional computation, return the average vector and the eigenspace of the given data.
    N, D = X_data.shape
    X_avg = X_data.mean(0)
    X_avgm = np.array([X_avg]*N)
    A = (X_data - X_avgm).T
    if mode == "high":
        S = A.dot(A.T) / N
        e_vals, e_vecs = np.linalg.eig(S)
    elif mode == "low":
        S = (A.T).dot(A) / N
        e_vals, e_vecs = np.linalg.eig(S)
        e_vecs = np.dot(A,e_vecs)
        e_vecs = e_vecs / np.linalg.norm(e_vecs, axis=0)
    return A, e_vals, e_vecs

def plot_image(face_vector, w, h, filename):
    # Reshape the given image data, plot the image
    plt.figure()
    image = np.reshape(np.absolute(face_vector),(w,h)).T
    fig = plt.imshow(image, cmap = 'gist_gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    plt.close()
    return

def plot_graph(type, eig_value, i, x, y, xtick, ytick, filename):
    # Plot the first i eigenvalues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if type == "bar":
        plt.bar(list(range(0, i)), eig_value[:i])
    else:plt.plot(list(range(0, i)), eig_value[:i])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick))
    plt.xlabel(x, fontsize=10)
    plt.ylabel(y, fontsize=10)
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    #plt.show(block=False)
    return


# KNN
def knn_classifier(rank, query_size, query_data, query_label,query_cam, gallery_size, gallery_data, gallery_cam, gallery_label):
    score = 0
    mAP = 0
    for q in range(query_size):
        cam_idx = [ idx for idx in range(gallery_size) if not( gallery_cam[idx] == query_cam[q] and gallery_label[idx] == query_label[q] )]    
        gallery_data_cam = gallery_data[cam_idx,:]
        gallery_label_cam = gallery_label[cam_idx]
        l2 = np.linalg.norm( (gallery_data_cam - query_data[q,:] ), ord=2, axis = 1)
        k_idx = np.argsort(l2)[:rank]
        ap = 0
        for idx in k_idx:
            if gallery_label_cam[idx] == query_label[q]:
                ap = ap + 1
        if ap > 0:
            score = score + 1
        mAP = mAP + ap / rank
    #print('knn_score = ', score / query_size)
    #print('mAP = ', mAP / query_size)
    return score/query_size, mAP/query_size

# Kmeans
def kmeans_classifier(class_size, query_size, query_data, query_label, gallery_size, gallery_data, gallery_label ):
    kmeans = KMeans(n_clusters=class_size, random_state=300).fit(gallery_data)
    cluster_idx = kmeans.labels_
    cluster_labels = np.zeros(class_size)
    for k in range(class_size):
        cluster_label_idx = [idx for idx in range(gallery_size) if cluster_idx[idx] == k]
        cluster_all_labels = gallery_label[cluster_label_idx]
        label = np.bincount(cluster_all_labels).argmax()
        cluster_labels[k] = label
    predict_kmeans = kmeans.predict(query_data)
    score = 0
    for q in range(query_size):
        if cluster_labels[predict_kmeans[q]] == query_label[q]:
            score = score + 1
    score = score/query_size
    #print('kmeans_score = ', score)
    return score