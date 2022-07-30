# -*- coding: utf-8 -*-
# EA-DBN-Surv  step2: use step1's Hyperparametric combination to train and predict EA-DBN-Surv model

import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.io import savemat,loadmat
from sklearn.metrics import r2_score
from itertools import combinations
import warnings
import torch
from utils.DBN import DBN
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
seed=0
if device == 'cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# Data processing

# loading data
data=pd.read_csv('rowdata.csv',header=0)
input_data=data.iloc[:,:-1].values
output_data=data.iloc[:,-1].values.reshape(-1,1)
# normalization
ss_X=MinMaxScaler(feature_range=(0,1)).fit(input_data)
ss_y=MinMaxScaler(feature_range=(0,1)).fit(output_data)
input_data = ss_X.transform(input_data)
output_data = ss_y.transform(output_data)
# divide data
x_train,x_valid, y_train, y_valid = train_test_split(input_data,output_data,test_size=0.3,random_state=0)



# Model construction and training

start_time=time.time()
input_length = x_train.shape[1]
output_length = y_train.shape[1]
# set hyperparameters (Random customized)
para=pd.read_excel('parabest.xls').iloc[0,1:].values
learning_rate=para[0]
learning_rate_finetune=para[1]
momentum=para[2]
epoch_pretrain = int(para[3])
epoch_finetune = int(para[4])
batch_size = int(para[5])
nlayer=int(para[6])
hidden_units = [int(i) for i in para[7:7+nlayer]]
tf='Sigmoid'
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD


# Build model
dbn = DBN(hidden_units, input_length, output_length,learning_rate=learning_rate,activate=tf, device=device)
n_parameters = sum(p.numel() for p in dbn.parameters() if p.requires_grad)
print(n_parameters)
dbn.pretrain(x_train, epoch=epoch_pretrain, batch_size=batch_size)
dbn.finetune(x_train, y_train, epoch_finetune, batch_size, loss_function,
             optimizer(dbn.parameters(), lr=learning_rate_finetune, momentum=momentum),
             validation=[x_valid,y_valid],shuffle=True,types=1)
# torch.save(dbn, 'utils/model.pkl')
dbn.train()
# Draw Loss Curve
plt.figure()
plt.plot(dbn.finetune_train_loss,c='dimgrey',label='训练误差')
plt.plot(dbn.finetune_valid_loss,c='silver',label='测试误差')
plt.legend()
plt.title('Loss Curve')
plt.savefig('Loss Curve.jpg')
plt.show()



# Model Prediction

y_predict = dbn.predict(x_valid, batch_size,types=1)
y_predict=y_predict.cpu().numpy()
end_time=time.time()
print('Time cost: %f s'%(end_time-start_time))

# Inverse normalization
test_pred = ss_y.inverse_transform(y_predict)
test_label = ss_y.inverse_transform(y_valid)

# draw fitting diagram of predicted value and real value
plt.figure()
plt.plot(test_label,c='dimgrey', label='真实值')
plt.plot(test_pred,c='silver',label='预测值')
plt.legend()
plt.show()

savemat('result/ga_dbn_result.mat',{'true':test_label,'pred':test_pred})


# Model evaluation
def bs(real,pred,name):
    # bs
    bs = np.mean(np.square(pred - real))
    print(name,' BS:', bs )

bs(y_valid,y_predict,'DBN')

def concordance_index(y_true, y_pred):
    possible_pairs = list(combinations(range(len(y_pred)), 2))
    concordance = 0
    permissible = 0
    for pair in possible_pairs:
        t1 = y_true[pair[0]]
        t2 = y_true[pair[1]]
        predicted_outcome_1 = y_pred[pair[0]]
        predicted_outcome_2 = y_pred[pair[1]]

        permissible = permissible + 1
        if t1 != t2:
            if t1 < t2:
                if predicted_outcome_1 < predicted_outcome_2:
                    concordance = concordance + 1
                    continue
                elif predicted_outcome_1 == predicted_outcome_2:
                    concordance = concordance + 0.5
                    continue
            elif t2 < t1:
                if predicted_outcome_2 < predicted_outcome_1:
                    concordance = concordance + 1
                    continue
                elif predicted_outcome_2 == predicted_outcome_1:
                    concordance = concordance + 0.5
                    continue
        elif t1 == t2:
            if predicted_outcome_1 == predicted_outcome_2:
                concordance = concordance + 1
                continue
            else:
                concordance = concordance + 0.5
                continue
    c = concordance / permissible

    return c

print("DBN--cindex",concordance_index(y_valid,y_predict))
