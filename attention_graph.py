import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

    
a = torch.randn(4, 2)
b = a.softmax(dim=1)
c = a.softmax(dim=0).transpose(0, 1)
print(a, '\n',  b, '\n', c)
d = b.matmul(c)
print(d)

d = d.numpy()

variables = ['A','B','C','X']
labels = ['ID_0','ID_1','ID_2','ID_3']

df = pd.DataFrame(d, columns=variables, index=labels)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
fig.colorbar(cax)

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

ax.set_xticklabels([''] + list(df.columns))
ax.set_yticklabels([''] + list(df.index))

plt.savefig("test.png")