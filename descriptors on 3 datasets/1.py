# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:14:04 2024

@author: admin
"""
import os
import pandas as pd
import numpy as np
import heapq
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix, accuracy_score

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
               'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
zscales = {'A': (0.07, -1.73, 0.09), 'V': (-2.69, -2.53, -1.29), 'L': (-4.19, -1.03, -0.98),
           'I': (-4.44, -1.68, -1.03), 'P': (-1.22, 0.88, 2.23), 'F': (-4.92, 1.3, 0.45),
           'W': (-4.75, 3.65, 0.85), 'M': (-2.49, -0.27, -0.41), 'K': (2.84, 1.41, -3.14),
           'R': (2.88, 2.52, -3.44), 'H': (2.41, 1.74, 1.11), 'G': (2.23, -5.36, 0.3),
           'S': (1.96, -1.63, 0.57), 'T': (0.92, -2.09, -1.4), 'C': (0.71, -0.97, 4.13),
           'Y': (-1.39, 2.32, 0.01), 'N': (3.22, 1.45, 0.84), 'Q': (2.18, 0.53, -1.14), 'D': (3.64, 1.13, 2.36), 'E': (3.08, 0.39, -0.07)}
VHSE = {'A': (0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48), 
        'C': (0.18, -1.67, -0.46, -0.21, 0.0, 1.2, -1.61, -0.19), 
        'D': (-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56), 
        'E': (-1.18, 0.4, 0.1, 0.36, -2.16, -0.17, 0.91, 0.02),
        'F': (1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.2), 
        'G': (-0.2, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34), 'H': (-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65), 'I': (1.27, -0.14, 0.3, -1.8, 0.3, -1.61, -0.16, -0.13), 'K': (-1.17, 0.7, 0.7, 0.8, 1.64, 0.67, 1.63, 0.13), 'L': (1.36, 0.07, 0.26, -0.8, 0.22, -1.37, 0.08, -0.62), 'M': (1.01, -0.53, 0.43, 0.0, 0.23, 0.1, -0.86, -0.68), 'N': (-0.99, 0.0, -0.37, 0.63, -0.55, 0.85, 0.75, -0.8), 'P': (0.22, -0.17, -0.5, 0.05, -0.01, -1.34, -0.19, 3.56), 'Q': (-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.2, -0.41), 'R': (-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.3, 0.83), 'S': (-0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11), 'T': (-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39), 'V': (0.76, -0.92, -0.17, -1.91, 0.22, -1.4, -0.24, 0.03), 'W': (1.5, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85), 'Y': (0.61, 1.6, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52)}
SVHEHS = {'A': (0.65, -5.37, -1.61, -1.82, -0.46, 0.44, -1.89, 0.0, 2.53, 9.12, 9.91, -0.8, -4.0), 
          'R': (-11.9, 6.82, 2.82, 4.48, -4.08, -0.22, 4.29, 0.05, -0.35, 3.88, -5.45, 1.8, 6.41), 'N': (-8.26, -0.55, 2.36, -0.28, 0.15, 0.03, 1.69, -0.19, -11.4, -0.99, -2.85, -5.03, 1.89), 
          'D': (-9.56, -0.88, 6.78, -2.49, 2.45, 0.26, 1.21, -0.15, -12.4, 5.31, -2.71, -1.19, -2.07), 
          'C': (7.08, -4.93, 0.47, 2.48, 3.63, -0.59, -1.65, 1.54, 2.03, -9.24, -3.27, -5.5, -10.8), 'Q': (-8.41, 1.03, 1.05, 0.32, -0.44, 1.0, 1.88, 0.82, -0.95, 5.3, -4.69, 0.99, -0.35), 'E': (-9.26, 0.43, 3.77, -4.56, -0.69, 0.64, 1.38, 1.24, -2.57, 16.03, -0.98, 1.27, -3.87), 'G': (-2.46, -7.21, -1.4, -4.0, -2.17, 0.37, -2.1, 0.74, -18.0, -6.73, 10.0, -7.69, 2.82), 'H': (-3.21, 1.47, 1.67, 2.42, -0.99, -1.41, 0.42, 0.14, 2.14, 0.29, -7.66, -3.62, -0.64), 'I': (12.46, -0.57, -3.79, -1.44, -0.52, 0.62, -1.08, -0.73, 13.53, -5.41, 5.26, 4.05, 1.36), 'L': (11.27, -0.36, -3.3, -1.27, -0.26, 0.9, -1.01, -1.75, 12.1, 4.57, 10.13, 1.3, 1.18), 'K': (-11.0, 3.46, 0.56, 2.01, -4.51, 0.36, 2.13, -1.62, -2.73, 10.16, -2.99, 0.33, 6.87), 'M': (7.49, -0.88, -0.24, 1.05, 1.46, 1.08, -0.97, -1.06, 13.35, 3.09, -3.33, -1.81, -4.09), 'F': (12.35, 2.81, -0.94, 2.4, 2.35, 0.43, -0.36, -0.27, 11.09, -4.08, -0.9, 0.38, 0.7), 'P': (-1.97, -1.3, -1.76, -1.08, 0.39, -6.4, -1.62, 1.75, -18.4, -5.85, -2.14, 14.8, -4.45), 'S': (-5.42, -3.62, 0.73, -0.49, 0.98, -0.21, -0.12, 0.68, -10.2, -2.03, 4.7, -1.8, 0.95), 'T': (-3.12, -2.47, -0.05, 0.02, 0.26, -0.07, 0.17, 0.07, -3.37, -4.17, 3.58, 0.19, 0.68), 'W': (10.99, 8.5, -1.93, 2.28, 1.83, 1.51, -0.22, -1.38, 9.84, -4.52, -10.5, -1.44, -0.09), 'Y': (3.73, 6.32, -0.92, 1.56, 1.25, 0.72, 0.24, 1.3, 2.65, -9.55, -5.39, 0.93, 6.59), 'V': (8.88, -2.71, -3.26, -1.6, -0.62, 0.54, -1.4, 0.3, 11.34, -5.1, 9.27, 2.86, 0.93)}
data = pd.read_csv(r"C:\Users\86156\OneDrive\桌面\Datasets\dataset1.csv", header=None)

folds = data.iloc[:, 2].tolist()
seqs = data.iloc[:, 0].tolist()
act = data.iloc[:, 1].tolist()

max_length = len(max(seqs, key=len))
aac_matrix = np.zeros((len(seqs), 20))
dipeptide_matrix = np.zeros((len(seqs), 400))
oht_matrix = np.zeros((len(seqs), max_length*20))

# onehot
def peptide_composition(seq):
    onehot = np.zeros(max_length * 20, dtype=int)
    for i, aa in enumerate(seq):
        aa_index = amino_acids.index(aa)
        onehot[i * 20 + aa_index] = 1
    return onehot

for i, seq in enumerate(seqs):
    oht_matrix[i] = peptide_composition(seq)

# dip
def dipeptide_composition(seq):
    dipeptide_comp = np.zeros(400)
    for i in range(len(seq)-1):
        dipeptide = seq[i]+seq[i+1]
        index1 = amino_acids.index(seq[i])
        index2 = amino_acids.index(seq[i+1])
        dipeptide_index = index1 * 20 + index2
        dipeptide_comp[dipeptide_index] += 1
    dipeptide_comp /= (len(seq) - 1) if len(seq) > 1 else 1
    return dipeptide_comp

for i, seq in enumerate(seqs):
    dipeptide_matrix[i] = dipeptide_composition(seq)

# aac
def aac_composition(seq):
    aac = np.zeros(20)
    for aa in seq:
        aac[amino_acids.index(aa)] += 1
    aac /= len(seq)
    return aac

for i, seq in enumerate(seqs):
    aac_matrix[i] = aac_composition(seq)
    
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
encoded_tripeptides = [''.join(huffman_codes[aa]
                               for aa in seq) for seq in seqs]
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
all_results = []
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

        result = np.concatenate((and_bits, or_bits, xor_bits, nand_bits, nor_bits, nxor_bits))
        results.append(result)
    all_results.append(results)

logi_matrix = np.concatenate(all_results, axis=0)
com_matrix = np.concatenate((oht_matrix, huffman_matrix, logi_matrix), axis=1)

# Zscales
def zscales_encoded(seq, zscales, max_length):
    encoded_matrix = []
    for aa in seq:
        if aa in zscales:
            encoded_row = zscales[aa]
        encoded_matrix.append(encoded_row)
    
    while len(encoded_matrix) < max_length:
        encoded_matrix.append([0] * len(encoded_row))
    
    zscales_matrix = np.array(encoded_matrix)
    zscales_matrix = np.concatenate(zscales_matrix, axis=0)
    return zscales_matrix

encoded_matrices = []
for seq in seqs:
    zscales_matrix = zscales_encoded(seq, zscales, max_length)
    encoded_matrices.append(zscales_matrix)
zscale_matrix = np.vstack(encoded_matrices)

# VHSE
def VHSE_encoded(seq, VHSE, max_length):
    encoded_matrix = []
    for aa in seq:
        if aa in VHSE:
            encoded_row = VHSE[aa]
        encoded_matrix.append(encoded_row)
    
    while len(encoded_matrix) < max_length:
        encoded_matrix.append([0] * len(encoded_row))
    
    vhse_matrix = np.array(encoded_matrix)
    vhse_matrix = np.concatenate(vhse_matrix, axis=0)
    return vhse_matrix

encoded_matrices = []
for seq in seqs:
    vhse_matrix = VHSE_encoded(seq, VHSE, max_length)
    encoded_matrices.append(vhse_matrix)
VHSE_matrix = np.vstack(encoded_matrices)
    
# SVHEHS
def SVHEHS_encoded(seq, SVHEHS, max_length):
    encoded_matrix = []
    for aa in seq:
        if aa in SVHEHS:
            encoded_row = SVHEHS[aa]
        encoded_matrix.append(encoded_row)
    
    while len(encoded_matrix) < max_length:
        encoded_matrix.append([0] * len(encoded_row))
    
    svhehs_matrix = np.array(encoded_matrix)
    svhehs_matrix = np.concatenate(svhehs_matrix, axis=0)
    return svhehs_matrix

encoded_matrices = []
for seq in seqs:
    svhehs_matrix = SVHEHS_encoded(seq, SVHEHS, max_length)
    encoded_matrices.append(svhehs_matrix)
SVHEHS_matrix = np.vstack(encoded_matrices)

######## model
def indicator(X, y, folds, descriptor_name):
    all_fpr = []       
    all_tpr = []      
    all_thresholds = []
    all_precision = []
    all_recall = []  
    metrics = []      

    for fold in range(5):
        train_indices = [i for i, fid in enumerate(folds) if fid != fold]
        test_indices = [i for i, fid in enumerate(folds) if fid == fold]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        svm_clf = svm.SVC(random_state=0, probability=True, class_weight='balanced')
        svm_clf.fit(X_train, y_train)
        
        y_pred = svm_clf.predict(X_test)
        y_prob = svm_clf.predict_proba(X_test)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics.append({
            'acc': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'sensitivity': recall_score(y_test, y_pred),
            'specificity': spe,
            'precision': precision_score(y_test, y_pred)
        })

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_thresholds.append(thresholds)
        all_precision.append(precision)
        all_recall.append(recall)

    mean_metrics = {
        metric: np.mean([m[metric] for m in metrics]) 
        for metric in metrics[0]
    }
    
    return {
        'mean_metrics': mean_metrics,
        'fpr_list': all_fpr,
        'tpr_list': all_tpr,
        'precision_list': all_precision,
        'recall_list': all_recall,
        'thresholds_list': all_thresholds,
        'descriptor': descriptor_name
    }

oht_indicators = indicator(oht_matrix, np.array(act), folds, 'OHT')
dip_indicators = indicator(dipeptide_matrix, np.array(act), folds, 'Dipeptide')
aac_indicators = indicator(aac_matrix, np.array(act), folds, 'AAC')
logi_indicators = indicator(logi_matrix, np.array(act), folds, 'LOGI')
huffman_indicators = indicator(huffman_matrix, np.array(act), folds, 'Huffman')
com_indicators = indicator(com_matrix, np.array(act), folds, 'COM')
zscales_indicators = indicator(zscale_matrix, np.array(act), folds, 'Z-scale')
VHSE_indicators = indicator(VHSE_matrix, np.array(act), folds, 'VHSE')
SVHEHS_indicators = indicator(SVHEHS_matrix, np.array(act), folds, 'SVHEHS')

def interpolate_curves(curve_data, x_type='fpr', num_points=100):
    x_uniform = np.linspace(0, 1, num_points)
    y_interpolated = []

    for i in range(5):
        x_raw = curve_data[f'{x_type}_list'][i]
        if x_type == 'fpr':
            y_raw = curve_data['tpr_list'][i]
        else: 
            y_raw = curve_data['precision_list'][i]
        
        if not np.all(np.diff(x_raw) >= 0):
            sort_indices = np.argsort(x_raw)
            x_raw = x_raw[sort_indices]
            y_raw = y_raw[sort_indices]
  
        x_raw, unique_indices = np.unique(x_raw, return_index=True)
        y_raw = y_raw[unique_indices]
        
        y_interp = np.interp(x_uniform, x_raw, y_raw)
        y_interpolated.append(y_interp)
    
    y_mean = np.mean(y_interpolated, axis=0)
    y_std = np.std(y_interpolated, axis=0)
    return x_uniform, y_mean, y_std

def save_interpolated_data(indicators, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    roc_x, roc_y_mean, roc_y_std = interpolate_curves(indicators, x_type='fpr')
    roc_df = pd.DataFrame({
        "FPR": roc_x,
        "TPR_Mean": roc_y_mean,
        "TPR_Std": roc_y_std
    })
    roc_filename = f"{indicators['descriptor']}_roc.csv"
    roc_path = os.path.join(output_dir, roc_filename)
    roc_df.to_csv(roc_path, index=False)

    prc_x, prc_y_mean, prc_y_std = interpolate_curves(indicators, x_type='recall')
    prc_df = pd.DataFrame({
        "Recall": prc_x,
        "Precision_Mean": prc_y_mean,
        "Precision_Std": prc_y_std
    })
    prc_filename = f"{indicators['descriptor']}_prc.csv"
    prc_path = os.path.join(output_dir, prc_filename)
    prc_df.to_csv(prc_path, index=False)

if __name__ == "__main__":
    all_indicators = [
        oht_indicators,
        dip_indicators,
        aac_indicators,
        logi_indicators,
        huffman_indicators,
        com_indicators,
        zscales_indicators,
        VHSE_indicators,
        SVHEHS_indicators
    ]
    for indicator in all_indicators:
        save_interpolated_data(indicator, output_dir="./curve_data")
        