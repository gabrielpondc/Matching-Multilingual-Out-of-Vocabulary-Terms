import sys, os
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot
import tqdm as tqdm
import gensim
import numpy as np
from pathlib import Path
import pickle
from time import time
from typing import List, Dict, Set, Tuple
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
model = Word2Vec.load('embeddings/k_model_300_30_50.pkl')
key_index=[116,
 113,
 90,
 9,
 110,
 106,
 89,
 10,
 115,
 104,
 34,
 25,
 75,
 8,
 74,
 35,
 103,
 11,
 108,
 109,
 112,
 54,
 105,
 107,
 87,
 32,
 45,
 55,
 46,
 56,
 13,
 42,
 88,
 58,
 86,
 98,
 43,
 12,
 114,
 47,
 111,
 92,
 72,
 59,
 97,
 85,
 99,
 60,
 38,
 31,
 26,
 44,
 36,
 37,
 50,
 101,
 53,
 84,
 52,
 102,
 68,
 33,
 21,
 22,
 19,
 27,
 29,
 20,
 78,
 94,
 6,
 49,
 51,
 100,
 30,
 18,
 67,
 76,
 7,
 66,
 73,
 48,
 41,
 63,
 79,
 91,
 1,
 83,
 40,
 64,
 23,
 4,
 77,
 28,
 65,
 81,
 14,
 17,
 3,
 61,
 15,
 39,
 96,
 2,
 82,
 5,
 62,
 80,
 95,
 24,
 93,
 69,
 71,
 16,
 70,
 57]
attrition = 'test.csv'
df_ac = pd.read_csv(attrition,encoding='utf-8')
df_ac
selected_columns = ['K','word']
df_attrition = df_ac.loc[:, selected_columns]

font_path = r'out.ttf'  # 此处改为自己字体的路径
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
X = model.wv[key_index]
pca = PCA(n_components=3)
result_pca= pca.fit_transform(X)
result = TSNE(n_components=3).fit_transform(result_pca)



# 可视化展示
fig=plt.figure(figsize=(200,200))
ax1=plt.axes(projection='3d')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')


ax1.scatter(result[:, 0], result[:, 1],result[:,2],cmap='blue',alpha=0.5, marker="o")
ax1.scatter(result[63, 0], result[63, 1],result[63,2],c='red',alpha=0.5, marker="o")
ax1.scatter(result[77, 0], result[77, 1],result[77,2],c='red',alpha=0.5, marker="o")

ad = list(key_index)
words=[]
for i in range(len(ad)):
	try:
		words.append(int(ad[i]))
	except ValueError:
		words.append(' ')
cword=[]
for i in words:
  try:
    a=df_attrition.loc[df_ac['K']==i]['word']
    cword.append(a.values[0])
  except:
	  cword.append('')
for i, word in zip(range(len(cword)),cword):
    ax1.text(result[i, 0], result[i, 1], result[i, 2], word)
    
#plt.savefig('embedding.svg', transparent=True, bbox_inches='tight', pad_inches=0.0)
plt.show()