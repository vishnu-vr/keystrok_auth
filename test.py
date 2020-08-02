import pandas as pd
import numpy as np

data = pd.read_csv('cleaned.csv')

features = np.array(data.drop('Label',axis=1))

# empty list
ret = []
# converting to list
data = features.tolist()
# finding length of rows
for i in range(len(data)):
	# concatinating columns
	# eg 0 -> 4 and 18 -> 20
	ret.append(data[0][0:4] + data[0][18:20])

# converting back to numpy arrays
ret_arr = np.array(ret)