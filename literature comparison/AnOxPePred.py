# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:28:58 2024

@author: admin
"""


import pandas as pd
import numpy as np
import heapq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

data = pd.read_csv(r"C:\Users\86156\OneDrive\桌面\Datasets\dataset1.csv", header=None)
seqs = data.iloc[:, 0].tolist()
act = data.iloc[:, 1].tolist()
folds = data.iloc[:, 2].tolist()

max_length = len(max(seqs, key=len)) 
oht_matrix = np.zeros((len(seqs), max_length * 20))

# onehot
def peptide_composition(seq):
    onehot = np.zeros(max_length * 20, dtype=int)
    for i, aa in enumerate(seq):
        aa_index = amino_acids.index(aa)
        onehot[i * 20 + aa_index] = 1
    return onehot

for i, seq in enumerate(seqs):
    oht_matrix[i] = peptide_composition(seq)

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
combined = np.concatenate((oht_matrix, huffman_matrix, logi_matrix), axis=1)

# AnOxPePred
folds = data.iloc[:, 2].tolist()

X = logi_matrix
y = np.array(act)
fold_ids = folds

auc_scores = []
f1_scores = []
mcc_scores = []
y_true_folds = []
y_pred_folds = []
y_prob_folds = []

params = {'learning_rate': [0.1],
          'max_iter': [50],
          'max_leaf_nodes': [30],
            'min_samples_leaf': [28],
            }

best_params = None
best_score = -1

for fold in range(5):
    train_indices = [i for i, fid in enumerate(fold_ids) if fid != fold]
    test_indices = [i for i, fid in enumerate(fold_ids) if fid == fold]

    X_train, X_test = pd.DataFrame([X[g] for g in train_indices]), pd.DataFrame([
        X[c] for c in test_indices])
    y_train, y_test = pd.DataFrame([y[g] for g in train_indices]).iloc[:,0], pd.DataFrame([
        y[c] for c in test_indices]).iloc[:,0]

    HGB = HistGradientBoostingClassifier(random_state=42)
    clf = GridSearchCV(estimator=HGB, param_grid=params, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    if clf.best_score_ > best_score:
        best_score = clf.best_score_
        best_params = clf.best_params_

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, (y_prob > 0.5).astype(int))

    auc_scores.append(auc)
    f1_scores.append(f1)
    mcc_scores.append(mcc)
    y_true_folds.append(y_test)
    y_pred_folds.append(y_pred)
    y_prob_folds.append(y_prob)

print('logi_matrix')
print("Average AUC:", np.mean(auc_scores))
print("Average F1:", np.mean(f1_scores))
print("Average MCC:", np.mean(mcc_scores))


X = huffman_matrix
y = np.array(act)
fold_ids = folds

auc_scores = []
f1_scores = []
mcc_scores = []
y_true_folds = []
y_pred_folds = []
y_prob_folds = []

params = {'learning_rate': [0.1],
          'max_iter': [50],
          'max_leaf_nodes': [30],
            'min_samples_leaf': [28],
            }

best_params = None
best_score = -1

for fold in range(5):
    train_indices = [i for i, fid in enumerate(fold_ids) if fid != fold]
    test_indices = [i for i, fid in enumerate(fold_ids) if fid == fold]

    X_train, X_test = pd.DataFrame([X[g] for g in train_indices]), pd.DataFrame([
        X[c] for c in test_indices])
    y_train, y_test = pd.DataFrame([y[g] for g in train_indices]).iloc[:,0], pd.DataFrame([
        y[c] for c in test_indices]).iloc[:,0]

    HGB = HistGradientBoostingClassifier(random_state=42)
    clf = GridSearchCV(estimator=HGB, param_grid=params, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    if clf.best_score_ > best_score:
        best_score = clf.best_score_
        best_params = clf.best_params_

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, (y_prob > 0.5).astype(int))

    auc_scores.append(auc)
    f1_scores.append(f1)
    mcc_scores.append(mcc)
    y_true_folds.append(y_test)
    y_pred_folds.append(y_pred)
    y_prob_folds.append(y_prob)

print('huffman_matrix')
print("Average AUC:", np.mean(auc_scores))
print("Average F1:", np.mean(f1_scores))
print("Average MCC:", np.mean(mcc_scores))

X = combined
y = np.array(act)
fold_ids = folds

auc_scores = []
f1_scores = []
mcc_scores = []
y_true_folds = []
y_pred_folds = []
y_prob_folds = []

params = {'learning_rate': [0.1],
          'max_iter': [50],
          'max_leaf_nodes': [30],
            'min_samples_leaf': [28],
            }

best_params = None
best_score = -1

for fold in range(5):
    train_indices = [i for i, fid in enumerate(fold_ids) if fid != fold]
    test_indices = [i for i, fid in enumerate(fold_ids) if fid == fold]

    X_train, X_test = pd.DataFrame([X[g] for g in train_indices]), pd.DataFrame([
        X[c] for c in test_indices])
    y_train, y_test = pd.DataFrame([y[g] for g in train_indices]).iloc[:,0], pd.DataFrame([
        y[c] for c in test_indices]).iloc[:,0]

    HGB = HistGradientBoostingClassifier(random_state=42)
    clf = GridSearchCV(estimator=HGB, param_grid=params, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    if clf.best_score_ > best_score:
        best_score = clf.best_score_
        best_params = clf.best_params_

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, (y_prob > 0.5).astype(int))

    auc_scores.append(auc)
    f1_scores.append(f1)
    mcc_scores.append(mcc)
    y_true_folds.append(y_test)
    y_pred_folds.append(y_pred)
    y_prob_folds.append(y_prob)

print('com_matrix')
print("Average AUC:", np.mean(auc_scores))
print("Average F1:", np.mean(f1_scores))
print("Average MCC:", np.mean(mcc_scores))