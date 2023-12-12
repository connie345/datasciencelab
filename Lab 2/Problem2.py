import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("DF2", index_col = 0)
cov = df.cov()
print(cov)
eigval,eigvec=np.linalg.eig(cov)
Q=np.dot(np.diag(1/np.sqrt(eigval)),eigvec.T)
white=Q@df.T
white = white.T
white.plot(x = 0, y = 1, kind= "scatter")
plt.show()

