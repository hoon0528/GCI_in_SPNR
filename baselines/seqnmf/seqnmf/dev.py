import seaborn as sns
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from seqnmf import seqnmf, plot

data = loadmat(os.path.join('data', 'MackeviciusData.mat'))
W, H, cost, loadings, power = seqnmf(data['NEURAL'])

h = plot(W, H)
h.show()

sns.heatmap(data['NEURAL'], cmap='gray_r')
plt.show()

