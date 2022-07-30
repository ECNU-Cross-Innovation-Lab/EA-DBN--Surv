from random_survival_forest import RandomSurvivalForest
from random_survival_forest import concordance_index
from lifelines import datasets
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
import math

# In[] 加载数据
rossi11=pd.read_csv('rowdata.csv',header=0)
rossi11['event']=1 # no censor
rossi=rossi11[:100]
y = rossi.loc[:, ["event", "week"]]
X = rossi.drop(["event", "week"], axis=1)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# model
print("RSF")
start_time = time.time()
rsf = RandomSurvivalForest(n_estimators=20, n_jobs=1, min_leaf=10)
rsf = rsf.fit(X, y)
print("--- %s seconds ---" % (time.time() - start_time))

# predict
y_pred = rsf.predict(X_test)
c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["event"])
print(c_val)

# try many times to have an average result(0.6483) for our for comparison
