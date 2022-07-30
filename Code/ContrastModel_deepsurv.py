
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(123)
df_train= pd.read_csv('rowdata.csv',header=0)
df_train['event']=1 # no censor
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)
df_train.head()

# Feature transforms

cols_standardize = ['age','grade','stage','total','examined','positive',
                    'size','extension','nodes','mets','total_mal','total_bor']
cols_leave = ['sex','first','race1','race2','history1','history2','surg_pri1',
              'surg_pri2','surg_pri3','surg_sco1','surg_sco2','surg_oth1',
              'surg_oth2','surg_oth3','source1','source2','source3','source4']
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
get_target = lambda df: (df['week'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val

# Neural net
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(n_parameters)

# Training the model
model = CoxPH(net, tt.optim.Adam)
batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
# plt.show()
lrfinder.get_best_lr()
print("best_lr: ",lrfinder.get_best_lr())
model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
# %%time
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
_ = log.plot()
# plt.show()
model.partial_log_likelihood(*val).mean()
# print("partial_log_likelihood: ",model.partial_log_likelihood(*val).mean())

# prediction

_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
# plt.show()

# Evaluation

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
ev.concordance_td()
print("c_index: ",ev.concordance_td())

time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()
# plt.show()
ev.integrated_brier_score(time_grid)
print("brier_score: ",ev.integrated_brier_score(time_grid))
