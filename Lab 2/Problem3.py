import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_uppercase



# # Part 1
year = input("Enter a year: ")
k = input("Enter the number of names: ")


df = pd.read_csv("Names/yob%i.txt" % int(year),delimiter=",",header=None)

namesWithS = df[df[df.columns[0]].str.startswith("S")]

namesWithS.sort_values(by=[df.columns[2]])

print(namesWithS[:int(k)])

# Part 2
df = pd.read_csv("Names/yob1880.txt",delimiter=",",header=None)

for index in range(1881,2016):
    tempDf = pd.read_csv("Names/yob%i.txt" % int(index),delimiter=",",header=None)
    df = pd.concat([df,tempDf])
    df = df.groupby([df.columns[0],df.columns[1]])[df.columns[2]].sum().reset_index()


name = input("Enter a name: ")


print("The number of men with the name %s is: %i" % (name,df.iloc[df.index[(df[df.columns[0]] == name) & (df[df.columns[1]] == "M")]][2]))
print("The number of men with the name %s is: %i" % (name,df.iloc[df.index[(df[df.columns[0]] == name) & (df[df.columns[1]] == "F")]][2]))

bestLetterCount, bestLetter = 0,"A"

for letter in ascii_uppercase:
    tempTotal = df[df[df.columns[0]].str.startswith(letter)][df.columns[2]].sum()
    if(tempTotal > bestLetterCount):
        bestLetter = letter
        bestLetterCount = tempTotal

print("The most common first letter of all names is %s with %i names" % (bestLetter,bestLetterCount))

# Part 3
name = input("Enter a name: ")

df = pd.read_csv("Names/yob1880.txt",delimiter=",",header=None)
rel_freq_female = []
rel_freq_male = []

for index in range(1880,2016):
    df = pd.read_csv("Names/yob%i.txt" % int(index),delimiter=",",header=None)
    female_count = df.iloc[df.index[(df[df.columns[0]] == name) & (df[df.columns[1]] == "F")]]
    male_count = df.iloc[df.index[(df[df.columns[0]] == name) & (df[df.columns[1]] == "M")]]
    if len(female_count) == 0:
        female_count = 0
    else:
        female_count = int(female_count[2])/df[df.columns[2]].sum()
    if len(male_count) == 0:
        male_count = 0
    else:
        male_count = int(male_count[2])/df[df.columns[2]].sum()
    rel_freq_female.append(female_count)
    rel_freq_male.append(male_count)

years = range(1880,2016)

plt.scatter(years,rel_freq_female,label="Female")
plt.scatter(years,rel_freq_male,label="Male")

plt.legend(loc="upper right")

plt.show()


# Part 4



# Part 5

year = input("Enter a year: ")

df = pd.read_csv("Names/yob%i.txt" % int(year),delimiter=",",header=None)
df2 = pd.read_csv("Names/yob%i.txt" % (int(year) - 1),delimiter=",",header=None)

dfGrouped = df.groupby(df.columns[0])[df.columns[2]].sum().reset_index()
df2Grouped = df2.groupby(df.columns[0])[df.columns[2]].sum().reset_index()

results = pd.DataFrame(columns=["Name","Frequency"])

for index, row in dfGrouped.iterrows():
    previous = df2Grouped.loc[df2Grouped[df2Grouped.columns[0]] == row[0]]
    if len(previous) == 0:
        previous = 0
    else:
        previous = int(previous[2])
    results = pd.concat([results,pd.DataFrame([[row[0], row[2]-previous]],columns=["Name","Frequency"])])

results = results.reset_index()

maxid = results["Frequency"].idxmax()

print("The name with the largest surge is %s." % results.iloc[maxid]["Name"])