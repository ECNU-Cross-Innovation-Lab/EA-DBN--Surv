# EA-DBN-Surv  step1: use EA to find best Hyperparametric combination

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from utils.DBN import DBN
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# definitions
def regress(x_train, y_train, x_valid, y_valid, num):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 0
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    learning_rate = num[0]
    learning_rate_finetune = num[1]
    momentum = num[2]
    epoch_pretrain = num[3]
    epoch_finetune = num[4]
    batch_size = num[5]
    nlayer = num[6]
    hidden_units = num[7:7 + nlayer]
    tf = 'Sigmoid'  # Sigmoid ReLU Tanh Softplus
    loss_function = torch.nn.MSELoss(reduction='mean')  # MSE
    optimizer = torch.optim.SGD  # use SGD algorithm
    input_length = x_train.shape[1]
    output_length = y_train.shape[1]

    # Build model
    dbn = DBN(hidden_units, input_length, output_length, learning_rate=learning_rate, activate=tf, device=device)
    # 1-pretrain
    dbn.pretrain(x_train, epoch=epoch_pretrain, batch_size=batch_size)
    # 2-fine tune
    dbn.finetune(x_train, y_train, epoch_finetune, batch_size, loss_function,
                 optimizer(dbn.parameters(), lr=learning_rate_finetune, momentum=momentum), shuffle=True, types=1)
    # predict
    y_predict = dbn.predict(x_valid, batch_size, types=1)
    y_predict = y_predict.cpu().numpy()  # --新

    F = np.mean(np.square((y_predict - y_valid)))  # --新
    return F


def get_fitness(x):  # get objective function
    return regress(x_train, y_train, x_valid, y_valid, num=x)


def select(pop, fitness):
    f = 1 / (fitness + 1e-3) # fitness function
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=f / f.sum()) #--从数组中随机抽取元素
    return pop[idx]


def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE_MAX).astype(np.bool_)  # --增加了_
        for i, point in enumerate(cross_points):
            if point == True and pop[i_, i] * parent[i] == 0:
                cross_points[i] = False
            if point == True and i == 6:
                cross_points[i] = False
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child, ub, lb):
    for point in range(DNA_SIZE_MAX):
        if np.random.rand() < MUTATION_RATE:
            if point <= 2:
                child[point] = lb[point] + (ub[point] - lb[point]) * np.random.rand()
            if 2 < point <= 5:
                child[point] = np.random.randint(lb[point], ub[point])
            #If the number of layers changes, clear the number of subsequent nodes first,
            # and then regenerate the number of nodes of each layer according to the new number of layers
            if point == 6:
                child[point] = np.random.randint(lb[point], ub[point])
                child[point + 1:] = 0
                for i in range(7, 7 + int(child[point])):
                    child[i] = np.random.randint(lb[-1], ub[-1])
            if point > 6:
                if child[point] != 0:
                    child[point] = np.random.randint(lb[-1], ub[-1])
    return child


# Data processing
data = pd.read_csv('rowdata.csv', header=0)
input_data = data.iloc[:, :-1].values
output_data = data.iloc[:, -1].values.reshape(-1, 1)
# normalization
ss_X = MinMaxScaler(feature_range=(0, 1)).fit(input_data)
ss_y = MinMaxScaler(feature_range=(0, 1)).fit(output_data)
input_data = ss_X.transform(input_data)
output_data = ss_y.transform(output_data)
# divide data
x_train, x_valid, y_train, y_valid = train_test_split(input_data, output_data, test_size=0.3, random_state=0)



# Model iteration

# Model configuration for EA
lb=[0.01,0.01,0.7, 1,  1,  32,1, 2]
ub=[ 0.1, 0.5,  1,21,201,257,4,61]
DNA_SIZE = 7
DNA_SIZE_MAX = DNA_SIZE + ub[-2] - 1
POP_SIZE = 10
CROSS_RATE = 0.5
MUTATION_RATE = 0.5
N_GENERATIONS = 50

# step1  Initialize population
pop_layers = np.zeros((POP_SIZE, DNA_SIZE))
for i in range(3):
    pop_layers[:, i] = lb[i] + (ub[i] - lb[i]) * np.random.rand(POP_SIZE, )
for i in range(3, 7):
    pop_layers[:, i] = np.random.randint(lb[i], ub[i], size=(POP_SIZE,))

pop = np.zeros((POP_SIZE, DNA_SIZE_MAX))

# --step2 Calculate fitness function of each chromosome
for i in range(POP_SIZE):
    pop_neurons = np.random.randint(lb[-1], ub[-1], size=(int(pop_layers[i][-1]),))
    pop_stack = np.hstack((pop_layers[i], pop_neurons))
    for j, gene in enumerate(pop_stack):
        pop[i][j] = gene

fitness = np.zeros([POP_SIZE, ])
for i in range(POP_SIZE):
    pop_list = list(pop[i])
    for j, each in enumerate(pop_list):
        if each == 0.0:
            pop_list = pop_list[:j]
    for k, each in enumerate(pop_list):
        pop_list[k] = each if k < 3 else int(each)
    fitness[i] = get_fitness(pop_list)
result = []
best_fitness = np.inf
trace = np.zeros([N_GENERATIONS, ])
for each_generation in range(N_GENERATIONS):

# -- step3 Pop evolution
    # 1.select  2.crossover  3.mutate
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child, ub, lb)
        parent = child

# --step4 Circularly calculate fitness and update population
    for i in range(POP_SIZE):
        pop_list = list(pop[i])
        for j, each in enumerate(pop_list):
            if each == 0.0:
                pop_list = pop_list[:j]
        for k, each in enumerate(pop_list):
            pop_list[k] = each if k < 3 else int(each)
        fitness[i] = get_fitness(pop_list)
    loc_pop = pop[np.argmin(fitness)]
    loc_fitness = np.min(fitness)
    if loc_fitness < best_fitness:
        best_fitness = loc_fitness.copy()
        best_pop = loc_pop.copy()
    trace[each_generation] = best_fitness  # --遍历
    print(each_generation + 1, '适应度值为:', best_fitness, ' 最优参数为:',
          [best_pop[i] if i <= 2 else int(best_pop[i]) for i in range(DNA_SIZE_MAX)])
    result.append([best_pop[i] if i <= 2 else int(best_pop[i]) for i in range(DNA_SIZE_MAX)])

# draw fitness curve
plt.figure()
plt.plot(trace)
plt.title('fitness curve')
plt.savefig('fitness curve.jpg')
plt.show()



# Output results

best = result[-1]
print('最优参数为：', best)
best = np.array(best).reshape(1, -1)
best_df = pd.DataFrame(best)
best_df.columns = ['预训练学习率', '微调学习率', '动量', '预训练次数', '微调次数', 'Batchsize', '隐含层层数', '隐含层节点数1', '隐含层节点数2', '隐含层节点数3']
best_df.to_excel('parabest.xls')
result = np.array(result)
trace = np.array(trace).reshape(-1, 1)
result = np.hstack([result, trace])
result_df = pd.DataFrame(result)
result_df.columns = ['预训练学习率', '微调学习率', '动量', '预训练次数', '微调次数', 'Batchsize', '隐含层层数', '隐含层节点数1', '隐含层节点数2', '隐含层节点数3',
                     '适应度值']
result_df.to_excel('result.xls')



