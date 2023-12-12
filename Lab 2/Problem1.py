import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Part 1

df = pd.read_csv("DF1",delimiter=",",header=None)

for index1 in range(len(df.columns)):
    tempFeature1 = df[df.columns[index1]]
    for index2 in range(index1+1,len(df.columns)):
        tempFeature2 = df[df.columns[index2]]
        plt.scatter(tempFeature1,tempFeature2)
        plt.xlabel("Feature %i" % index1)
        plt.ylabel("Feature %i" % index2)
        plt.savefig("Feature_%i_Feature_%i_Scatter.png" % (index1,index2))
        plt.clf()


# Part 2

print(df.cov())


# Part 3

vals = np.random.multivariate_normal([0,0,0],[[1,0,0],[0,1,1],[0,1,1]],1500)


terms = []
results = []

for i in range(1,len(vals)):
    temp = pd.DataFrame(vals[:i])
    tempCov = temp.cov()
    results.append(tempCov[1][2])
    terms.append(i)

plt.scatter(terms,results)
plt.ylabel("Estimate Correlation")
plt.xlabel("Number of Samples")
plt.show()