import torch
import warnings
import torch.nn as nn
import numpy as np

from .RBM import RBM
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim import Adam, SGD

import torch.nn.functional as F


class DBN(nn.Module):
    def __init__(self, hidden_units, visible_units=31, output_units=1, k=1,
                 learning_rate=1e-5, learning_rate_decay=False, activate='Sigmoid',
                 increase_to_cd_k=True, device='cpu'):
        super(DBN, self).__init__()

        self.n_layers = len(hidden_units)
        self.rbm_layers = []
        self.rbm_nodes = []
        self.device = device
        self.is_pretrained = False
        self.is_finetune = False
        tf = {'Sigmoid': torch.nn.Sigmoid(), 'Softplus': torch.nn.Softplus(),
              'ReLU': torch.nn.ReLU(), 'Tanh': torch.nn.Tanh()}
        self.activation = tf[activate]  # Tanh Softplus ReLU Sigmoid

        # Creating different RBM layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_units[i - 1]
            rbm = RBM(visible_units=input_size, hidden_units=hidden_units[i],
                      k=k, learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      increase_to_cd_k=increase_to_cd_k, device=device)

            self.rbm_layers.append(rbm)

        self.W_rec = [self.rbm_layers[i].weight for i in range(self.n_layers)]
        self.bias_rec = [self.rbm_layers[i].h_bias for i in range(self.n_layers)]

        for i in range(self.n_layers):
            self.register_parameter('W_rec%i' % i, self.W_rec[i])
            self.register_parameter('bias_rec%i' % i, self.bias_rec[i])

        self.bpnn = torch.nn.Linear(hidden_units[-1], output_units).to(self.device)

    def forward(self, input_data):
        """
        running a single forward process.

        Args:
            input_data: Input data of the first RBM layer. Shape:
                [batch_size, input_length]

        Returns: Output of the last RBM hidden layer.

        """

        v = input_data.to(self.device)
        hid_output = v.clone()
        for i in range(len(self.rbm_layers)):
            #            hid_output, _ = self.rbm_layers[i].to_hidden(hid_output)
            hid_output = torch.matmul(hid_output, self.W_rec[i]) + self.bias_rec[i]
            hid_output = self.activation(hid_output)
        self.feature = hid_output
        output = self.bpnn(hid_output)
        return output

    def reconstruct(self, input_data):
        """
        Go forward to the last layer and then go feed backward back to the
        first layer.

        Args:
            input_data: Input data of the first RBM layer. Shape:
                [batch_size, input_length]

        Returns: Reconstructed output of the first RBM visible layer.

        """
        h = input_data.to(self.device)
        p_h = 0
        for i in range(len(self.rbm_layers)):
            # h = h.view((h.shape[0], -1))
            p_h, h = self.rbm_layers[i].to_hidden(h)  # 一直前进

        for i in range(len(self.rbm_layers) - 1, -1, -1):
            # h = h.view((h.shape[0], -1))
            p_h, h = self.rbm_layers[i].to_visible(h)  # 一直返回
        return p_h, h

    def pretrain(
            self, x, epoch=50, batch_size=10):
        """
        Train the DBN model layer by layer and fine-tuning with regression
        layer.

        Args:
            x: DBN model input data. Shape: [batch_size, input_length]
            epoch: Train epoch for each RBM.
            batch_size: DBN train batch size.

        Returns:

        """
        hid_output_i = torch.tensor(x, dtype=torch.float, device=self.device)

        for i in range(len(self.rbm_layers)):
            # print("Training rbm layer {}.".format(i + 1))

            dataset_i = TensorDataset(hid_output_i)
            dataloader_i = DataLoader(dataset_i, batch_size=batch_size, drop_last=False)

            self.rbm_layers[i].train_rbm(dataloader_i, epoch)
            hid_output_i, _ = self.rbm_layers[i].forward(hid_output_i)

        # Set pretrain finish flag.   ？？？？？
        self.is_pretrained = True
        return

    def finetune(self, x, y, epoch, batch_size, loss_function, optimizer, validation=None, shuffle=False, types=0):
        """
        Fine-tune the train dataset.

        Args:
            x: Input data
            y: Target data
            epoch: Fine-tuning epoch
            batch_size: Train batch size
            loss_function: Train loss function
            optimizer: Finetune optimizer
            shuffle: True if shuffle train data
            types : 0 for classification, 1 for regression
        Returns:

        """

        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)
        x = torch.tensor(x).type(torch.FloatTensor)
        y = torch.tensor(y).type(torch.LongTensor) if types == 0 else torch.tensor(y).type(torch.FloatTensor)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=False)
        self.finetune_train_loss = []
        if validation is not None:
            x = torch.tensor(validation[0]).type(torch.FloatTensor)
            y = torch.tensor(validation[1]).type(torch.LongTensor) if types == 0 else torch.tensor(validation[1]).type(
                torch.FloatTensor)
            dataset = TensorDataset(x, y)
            dataloader_val = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
            self.finetune_valid_loss = []

        # print('Begin fine-tuning.')
        for epoch_i in range(1, epoch + 1):
            total_loss = 0
            for batch in dataloader:
                input_data, ground_truth = batch
                input_data = input_data.to(self.device)
                ground_truth = ground_truth.to(self.device)
                output = self.forward(input_data)
                loss = loss_function(output, ground_truth)   ##### lossfunction
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.finetune_train_loss.append(total_loss)
            #            print(('Epoch:{0}/{1} -rbm_train_loss: ' + disp).format(epoch_i, epoch, total_loss))
            if validation is not None:
                total_loss = 0
                with torch.no_grad():
                    for batch in dataloader_val:
                        input_data, ground_truth = batch
                        input_data = input_data.to(self.device)
                        ground_truth = ground_truth.to(self.device)
                        output = self.forward(input_data)
                        loss = loss_function(output, ground_truth)
                        total_loss += loss.item()

                self.finetune_valid_loss.append(total_loss)
        #            print(('Epoch:{0}/{1} -rbm_valid_loss: ' + disp).format(epoch_i, epoch, total_loss))

        self.is_finetune = True

        return

    def predict(self, x, batch_size, shuffle=False, types=0):
        """
        Predict

        Args:
            x: DBN input data. Type: ndarray. Shape: (batch_size, visible_units)
            batch_size: Batch size for DBN model.
            shuffle: True if shuffle predict input data.
            types=0
        Returns: Prediction result. Type: torch.tensor().
            Device is 'cpu' so it can transferred to ndarray.
            Shape: (batch_size, output_units)
        """
        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        if not self.is_pretrained:
            warnings.warn("Hasn't finetuned DBN model yet. Recommend "
                          "run self.finetune() first.", RuntimeWarning)
        y_predict = torch.tensor([])

        x_tensor = torch.tensor(x, dtype=torch.float, device=self.device)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size, shuffle)
        with torch.no_grad():
            for batch in dataloader:
                y = self.forward(batch[0])
                y_predict = torch.cat((y_predict, y.cpu()), 0)
        if types == 0:
            _, predicted_labels = torch.max(y_predict, 1)

            return predicted_labels, F.softmax(y_predict, dim=1)
        elif types == 1:
            return y_predict


class FineTuningDataset(Dataset):
    """
    Dataset class for whole dataset. x: input data. y: output data
    """

    def __init__(self, x, y):
        #        self.x = x.astype(np.float32)
        #        self.y = y.astype(np.int32)
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
