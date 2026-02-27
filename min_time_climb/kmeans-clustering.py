
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
import os

cmap = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red'}

def plot_case(i, group):
    # fufile = os.path.abspath('./{} Result Data/{} Compression F-u Data.csv'.format(basename, basename))")
	fufile = os.path.join(os.path.dirname(__file__), f"Data_Files/data_case_{i}.csv")
	with open(fufile, 'r') as f:
		tf = pd.read_csv(f)
	tf.plot(x='time', y='h', ax=plt.gca(), legend=[str(i)], c=cmap[int(group)])
    
    

with open('./extracted_features.csv', 'r') as f:
    extracted_features = pd.read_csv(f)

scaler = MinMaxScaler()
scaler.fit(extracted_features)
X = scaler.transform(extracted_features)


# inertia = []
# for i in range(1, 11):
#     kmeans = KMeans(
#         n_clusters=i, init="k-means++",
#         n_init=10, tol=1e-4, 
#         random_state=42
#     )
#     kmeans.fit(X)
#     inertia.append(kmeans.inertia_)

# plt.plot(list(range(1,11)), inertia)

## create kmeans with 3 clusters

kmeans = KMeans(
    n_clusters=3, init="k-means++",
    n_init=10, tol=1e-4, 
    random_state=42
)
kmeans.fit(X)
labels = kmeans.labels_

plt.figure()
with open('outputs-3.txt', 'w') as f:
	f.write('Test Case, Group')
	for test, label in zip(itertools.chain(range(1, 38)), labels):
		f.write("{},{}\n".format(test, label))
		plot_case(test, label)

kmeans = KMeans(
    n_clusters=4, init="k-means++",
    n_init=10, tol=1e-4, 
    random_state=42
)
kmeans.fit(X)
labels = kmeans.labels_

plt.figure()
with open('outputs-4.txt', 'w') as f:
	f.write('Test Case, Group')
	for test, label in zip(itertools.chain(range(1, 38)), labels):
		f.write("{},{}\n".format(test, label))
		plot_case(test, label)

plt.show()