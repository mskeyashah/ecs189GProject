'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Evaluate_Precision import Evaluate_Precision
from code.stage_5_code.Evaluate_Recall import Evaluate_Recall
from code.stage_5_code.Evaluate_F1 import Evaluate_F1
from matplotlib import pyplot

import torch
from torch import nn
import torch.optim as optim
import numpy as np

def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(A.shape[0])
    A_hat = A + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)


class Method_GCN_Cora(method, nn.Module):
    data = None

    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.01

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        input_dim = 1433
        hidden_dim = 16
        num_classes = 7
        p = 0.5

        self.gcn_layer1 = nn.Linear(input_dim, hidden_dim)
        self.acti1 = nn.ReLU(inplace=True)
        self.gcn_layer2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p)
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, A, X):
        '''Forward propagation'''
        A = torch.from_numpy(preprocess_adj(A)).float()
        X = self.dropout(X.float())
        F = torch.mm(A, X)
        F = self.gcn_layer1(F)
        F = self.acti1(F)
        F = self.dropout(F)
        F = torch.mm(A, F)
        output = self.gcn_layer2(F)
        return output

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy(' ', '')
        precision_evaluator = Evaluate_Precision(' ', '')
        recall_evaluator = Evaluate_Recall(' ', '')
        f1_evaluator = Evaluate_F1(' ', '')
        loss = []
        # it will be an iterative gradient updating process
        y = y[self.data['train_mask']]
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(self.data['adj'], X)
            # convert y to torch.tensor as well
            # calculate the training loss
            y_pred = y_pred[self.data['train_mask']]
            train_loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%5 == 0:
                accuracy_evaluator.data = {'true_y': y.max(1)[1], 'pred_y': y_pred.max(1)[1]}
                precision_evaluator.data = {'true_y': y.max(1)[1], 'pred_y': y_pred.max(1)[1]}
                recall_evaluator.data = {'true_y': y.max(1)[1], 'pred_y': y_pred.max(1)[1]}
                f1_evaluator.data = {'true_y': y.max(1)[1], 'pred_y': y_pred.max(1)[1]}

                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Precision:',
                    precision_evaluator.evaluate(), 'Recall:', recall_evaluator.evaluate(), 'F1:',
                    f1_evaluator.evaluate(), 'Loss:', train_loss.item())
            loss.append(train_loss.item())
        pyplot.plot(loss)
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss Value')
        pyplot.title('Epochs vs. Loss')
        pyplot.savefig('../../result/stage_5_result/loss_plot'+self.method_name+'.png')
        pyplot.clf()

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(self.data['adj'],X)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('--start training...')
        self.train(self.data['features'], self.data['y_train'])
        print('--start testing...')
        pred_y = self.test(self.data['features'])
        pred_y = pred_y[self.data['test_mask']]
        y = self.data['y_test'][self.data['test_mask']].max(1)[1]
        accuracy_evaluator = Evaluate_Accuracy(' ', '')
        precision_evaluator = Evaluate_Precision(' ', '')
        recall_evaluator = Evaluate_Recall(' ', '')
        f1_evaluator = Evaluate_F1(' ', '')

        accuracy_evaluator.data = {'true_y': y, 'pred_y': pred_y}
        precision_evaluator.data = {'true_y': y, 'pred_y': pred_y}
        recall_evaluator.data = {'true_y': y, 'pred_y': pred_y}
        f1_evaluator.data = {'true_y': y, 'pred_y': pred_y}

        print('Accuracy:', accuracy_evaluator.evaluate(), 'Precision:', precision_evaluator.evaluate(),
              'Recall:', recall_evaluator.evaluate(), 'F1:', f1_evaluator.evaluate())
        return {'pred_y': pred_y, 'true_y': y}
