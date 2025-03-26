# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:50:34 2024

@author: admin
"""

import pandas as pd
import numpy as np
import heapq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

data = pd.read_csv(r"C:/Users/li/Desktop/444.csv", header=None)
seqs = data.iloc[:, 0].tolist()
act = data.iloc[:, 1].tolist()

max_length = len(max(seqs, key=len))
aac_matrix = np.zeros((len(seqs), 20)) 
dipeptide_matrix=np.zeros((len(seqs), 400))  
peptide_matrix = np.zeros((len(seqs), max_length * 20))

def peptide_composition(seq):
    onehot = np.zeros(max_length * 20, dtype=int)
    for i, aa in enumerate(seq):
        aa_index = amino_acids.index(aa)
        onehot[i * 20 + aa_index] = 1
    return onehot

for i, seq in enumerate(seqs):
    peptide_matrix[i] = peptide_composition(seq)

# huffman
def frequency(sequences):
    fre = {aa: 0 for aa in amino_acids}
    total_length = 0

    for seq in sequences:
        for aa in seq:
            if aa in fre:
                fre[aa] += 1
        total_length += len(seq)

    freq_dict = {}
    for aa, count in sorted(fre.items(), key=lambda item: item[1], reverse=False):
        percentage = (count / total_length) * 100 if total_length > 0 else 0
        freq_dict[aa] = (count, percentage)

    return freq_dict

freq_dict = frequency(seqs)

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_huffman_codes(root, prefix="", codebook={}):
    if root is not None:
        if root.symbol is not None:
            codebook[root.symbol] = prefix
        generate_huffman_codes(root.left, prefix + "0", codebook)
        generate_huffman_codes(root.right, prefix + "1", codebook)
    return codebook

huffman_tree = build_huffman_tree(freq_dict)
huffman_codes = generate_huffman_codes(huffman_tree)

encoded_tripeptides = [''.join(huffman_codes[aa] for aa in seq) for seq in seqs]

num_samples = len(encoded_tripeptides)
huffman_matrix = np.zeros((num_samples, max_length * 20), dtype=int)

for i, encoding in enumerate(encoded_tripeptides):
    huffman_matrix[i, -len(encoding):] = [int(bit) for bit in encoding]

# logi
peptide_matrices = []   
results = []  
for seq in seqs:  
    result = peptide_composition(seq)  
    results.append(result)  
peptide_matrices.append(results)  

all_results=[]
for peptide_matrice in peptide_matrices:
    results = []
    for matrice in peptide_matrice:
        seq_parts = [matrice[i:i+20] for i in range(0, matrice.shape[0], 20)]
    
        and_bits = np.bitwise_and.reduce(seq_parts)
        or_bits = np.bitwise_or.reduce(seq_parts)
        xor_bits = np.bitwise_xor.reduce(seq_parts)
        nand_bits = -((and_bits) - 1)
        nor_bits = -((or_bits) - 1)
        nxor_bits = -((xor_bits) - 1)
    
        result = np.concatenate((and_bits, or_bits, xor_bits, nand_bits, nor_bits, nxor_bits ))
        results.append(result)
    all_results.append(results)

logi_matrix = np.concatenate(all_results, axis=0)
combine = np.concatenate((peptide_matrix, huffman_matrix, logi_matrix), axis=1)

# SVM
def SVM_opt(X, y, length):
    y_mean = np.full((1,length), np.sum(y)/length)
    count = -1
    global verify
    verify = np.zeros((576,4))
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    for first_iteration in range(-1,7):
        c = 2**first_iteration
        for second_iteration in range(-8,0):
            p = 2**second_iteration
            for third_iteration in range(-8,1):
                g = 2**third_iteration 
                YPred = np.zeros(length)        
                count = count + 1
                for train_index, test_index in loo.split(X):      
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]             
                    regr = svm.SVR(kernel ='rbf',gamma = g , coef0 = 0.0,
    		    tol = 0.001, C = c, epsilon = p, shrinking = True, cache_size = 40,		
    		    verbose = False,max_iter = -1)
                    regr = regr.fit(X_train, y_train)
                    y_pre = regr.predict(X_test)
                    YPred[test_index] = y_pre
                global num_opt
                num_opt = (1-np.sum((y-YPred)**2)/np.sum((y-y_mean)**2))
                verify[count][0] = c
                verify[count][1] = p
                verify[count][2] = g
                verify[count][3] = num_opt
    opt = verify[np.argsort(verify[:,3])]
    g_opt, c_opt, p_opt = opt[-1,2], opt[-1,0], opt[-1,1]
    regr = svm.SVR(kernel ='rbf',gamma = g_opt , coef0 = 0.0, tol = 0.001, C = c_opt,
                    epsilon = p_opt, shrinking = True, cache_size = 40,		
    		    verbose = False,max_iter = -1)
    SVM_reg = regr.fit(X, y)
    Y_predication = SVM_reg.predict(X)
    return Y_predication, opt[-1,3], g_opt, c_opt, p_opt

# random_forest
def RF(X, y, length):
    Q2_array = np.zeros((101,2))
    for n in range(1,101):
        clf = RandomForestRegressor(n_estimators = n, random_state = 0,n_jobs=-1)
        y_mean = np.full((1,length), np.sum(y)/length)
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        YPred = np.zeros(length)
        for train_index, test_index in loo.split(X):      
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = clf.fit(X_train, y_train)
            y_pre = clf.predict(X_test)
            YPred[test_index] = y_pre
        Q2 = (1-np.sum((y - YPred)**2)/np.sum((y-y_mean)**2))
        Q2_array[n][0] = Q2
        Q2_array[n][1] = n
    opt = Q2_array[np.argsort(Q2_array[:,0])]
    n_forest = opt[-1, 1]
    clf = RandomForestRegressor(n_estimators = int(n_forest), random_state = 0, n_jobs=-1)
    clf = clf.fit(X,y)
    Y_predication = clf.predict(X)
    return Y_predication, opt[-1,0], int(n_forest)

GradientBoostingRegressor
def GBR(X, y, length):
    Q2_array = np.zeros((1001, 2))
    for n in range(1, 1001):
        regr = GradientBoostingRegressor(random_state=0, n_estimators=n)
        y_mean = np.full((1,length), np.sum(y)/length)
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        YPred = np.zeros(length) 
        for train_index, test_index in loo.split(X):      
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index] 
            AdB_reg = regr.fit(X_train, y_train)
            y_pre = AdB_reg.predict(X_test)
            YPred[test_index] = y_pre
        Q2 = (1-np.sum((y - YPred)**2)/np.sum((y-y_mean)**2))
        Q2_array[n][0] = Q2
        Q2_array[n][1] = n    
    opt = Q2_array[np.argsort(Q2_array[:,0])]
    n_estimator = opt[-1, 1]
    clf = GradientBoostingRegressor(random_state=0, n_estimators = int(n_estimator))
    clf.fit(X, y)
    Y_predication = clf.predict(X)
    return Y_predication, opt[-1, 0], int(n_estimator)

# MLP
def MLP(X, y, length):
    regr = MLPRegressor(random_state=1, max_iter=500)
    y_mean = np.full((1,length), np.sum(y)/length)
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    YPred = np.zeros(length) 
    for train_index, test_index in loo.split(X):      
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        AdB_reg = regr.fit(X_train, y_train)
        y_pre = AdB_reg.predict(X_test)
        YPred[test_index] = y_pre
    Q2 = (1-np.sum((y - YPred)**2)/np.sum((y-y_mean)**2))
    regr.fit(X, y)
    Y_prediction = regr.predict(X)
    return Y_prediction, Q2, YPred

# indicators
def Indicators(y_hat, y, length):
    MAE = np.mean(np.abs(y - y_hat))
    MSE = np.mean(np.square(y - y_hat))
    RMSE = np.sqrt(np.mean(np.square(y - y_hat)))
    MAPE = np.mean(np.abs((y - y_hat) / y)) * 100
    y_mean = np.full((1, length), np.sum(y)/ length)
    R2 = (1-np.sum((y - y_hat)**2)/np.sum((y-y_mean)**2))
    return MAE,MSE,RMSE,MAPE,R2

length = len(seqs)

SVM_opt_result, SVM_Q2, g_opt, c_opt, p_opt = SVM_opt(logi_matrix, np.array(act), length)
SVM_indicators = Indicators(SVM_opt_result, np.array(act), length)

RF_result, RF_Q2, RF_n = RF(logi_matrix, np.array(act), length)
RF_indicators = Indicators(RF_result, np.array(act), length)

GBR_result, GBR_Q2, GBR_n = GBR(logi_matrix, np.array(act), length)
GBR_indicators = Indicators(GBR_result, np.array(act), length)

MLP_result, MLP_Q2, MLP_YPred = MLP(logi_matrix, np.array(act), length)
MLP_indicators = Indicators(MLP_result, np.array(act), length)