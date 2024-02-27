import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Aufgabe 1
df = pd.read_csv('iris.csv')
print(df.head())
print(df.tail())
print(df.info())
print("#"*50)
#Aufgabe 2

new_frame = df[["sepal.length", "sepal.width"]]

print(new_frame.head(10))

setosa = df[df["species"] == "Setosa"]
versicolor1 = df[df["species"] == "Versicolor"]
virginica = df[df["species"] == "Virginica"]

print(setosa)

versicolor = df[(df["petal.length"]>1.5) & (df["species"] == "Versicolor")]

print(versicolor)
print("#"*50)


#Aufgabe 3
df["sepal_area"] = df["sepal.length"] * df["sepal.width"]
print(df.head())

replacements = {'Setosa': 'S', 'versicolor': 'Ve', "virginica" : "Vi"}

df['species'] = df['species'].replace(replacements)

print(df.head(10))

df = df.drop(df[df['sepal.length'] <4.5].index)

print(df.head(10))

print("#"*50)

#plotter

#plt.scatter(df["sepal.length"], df["sepal.width"])
#plt.xlabel('sepal.length')
#plt.ylabel('sepal.width')
#plt.show()


plt.scatter(setosa['sepal.length'], setosa['sepal.width'], color = 'red', marker='s')

plt.scatter(versicolor['sepal.length'], versicolor['sepal.width'], color = 'blue', marker='o')

plt.scatter(virginica['sepal.length'], virginica['sepal.width'], color = 'green',marker='^')

plt.xlabel('sepal.length')

plt.ylabel('sepal.width')

plt.show()

plt.show()