import matplotlib.pyplot as plt
import pandas as pd
import os

ids = range(25, 26)
plt.figure()
for i in ids:
    fufile = os.path.join(os.path.dirname(__file__), f"Data_Files/data_case_{i}.csv")

    # fufile = os.path.abspath('./{} Result Data/{} Compression F-u Data.csv'.format(basename, basename))
    with open(fufile, 'r') as f:
        tf = pd.read_csv(f)
    tf.plot(x='time', y='h', ax=plt.gca(), legend=["h_"+str(i)])

plt.show()