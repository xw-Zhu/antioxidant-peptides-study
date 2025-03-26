# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:21:18 2024

@author: admin
"""

import numpy as np
import pandas as pd
import heapq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

data = pd.read_csv(r"C:/Users/86156/OneDrive/桌面/444.csv",header=None)
seqs = data.iloc[:, 0].tolist()
act = data.iloc[:, 1].tolist()

max_length = len(max(seqs, key=len))
aac_matrix = np.zeros((len(seqs), 20)) 
oht_matrix = np.zeros((len(seqs),20*max_length))  
dipeptide_matrix=np.zeros((len(seqs), 400))  

###### onehot
def peptide_composition(seq):
    onehot = np.zeros(max_length * 20, dtype=int)
    for i, aa in enumerate(seq):
        aa_index = amino_acids.index(aa)
        onehot[i * 20 + aa_index] = 1
    return onehot

for i, seq in enumerate(seqs):
    oht_matrix[i] = peptide_composition(seq)

###### dipeptide
def dipeptide_composition(seq):  
    dipeptide_comp=np.zeros(400)  
    for i in range(len(seq)-1):  
       dipeptide = seq[i]+seq[i+1]  
       index1 = amino_acids.index(seq[i])  
       index2 = amino_acids.index(seq[i+1])  
       dipeptide_index = index1 * 20 + index2  
       dipeptide_comp[dipeptide_index] += 1  
    dipeptide_comp /= (len(seq) - 1) if len(seq) > 1 else 1 
    return dipeptide_comp  

for i,seq in enumerate(seqs):  
    dipeptide_matrix[i] = dipeptide_composition(seq)
    
###### amino acid
def aac_composition(seq):  
    aac = np.zeros(20)  
    for aa in seq:  
        aac[amino_acids.index(aa)]+=1  
    aac /= len(seq) 
    return aac  
  
for i,seq in enumerate(seqs):  
    aac_matrix[i] = aac_composition(seq)

###### logi
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
        nots = [-((seq_part) - 1) for seq_part in seq_parts]
        nots_bits = np.concatenate(nots, axis=0)
    
        result = np.concatenate((and_bits, or_bits, xor_bits, nand_bits, nor_bits, nxor_bits, nots_bits))
        results.append(result)
    all_results.append(results)

logi_matrix = np.concatenate(all_results, axis=0)

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

combined = np.concatenate((huffman_matrix,logi_matrix,oht_matrix),axis=1)

# aac
X = aac_matrix  
y = act  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
RF = RandomForestRegressor(random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"aac(MSE): {mse}")
print(f"aacR²: {r2}")

#dipeptide
X = dipeptide_matrix  
y = act  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
RF = RandomForestRegressor(random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"dip(MSE): {mse}")
print(f"dipR²: {r2}")

# onehot
X = oht_matrix  
y = act  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
RF = RandomForestRegressor(random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"oht(MSE): {mse}")
print(f"ohtR²: {r2}")

#logical
X = logi_matrix
y = act  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
RF = RandomForestRegressor(random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"logi(MSE): {mse}")
print(f"logiR²: {r2}")

# huffman
X = huffman_matrix
y = act  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
RF = RandomForestRegressor(random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"huffman(MSE): {mse}")
print(f"huffmanR²: {r2}")

# whole
X = combined
y = act  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
RF = RandomForestRegressor(random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"combined(MSE): {mse}")
print(f"combinedR²: {r2}")
