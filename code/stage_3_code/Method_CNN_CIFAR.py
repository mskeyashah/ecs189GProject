'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from torch.nn import Sequential


from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Evaluate_Precision import Evaluate_Precision
from code.stage_3_code.Evaluate_Recall import Evaluate_Recall
from code.stage_3_code.Evaluate_F1 import Evaluate_F1
from matplotlib import pyplot

import torch
from torch import nn
import torch.optim as optim
import numpy as np


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.05

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))


    def forward(self, x):
        '''Forward propagation'''
        return self.network(x)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
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
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself

        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true =  torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()

            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%10 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                precision_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                recall_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                f1_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}

                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Precision:',
                    precision_evaluator.evaluate(), 'Recall:', recall_evaluator.evaluate(), 'F1:',
                    f1_evaluator.evaluate(), 'Loss:', train_loss.item())
            loss.append(train_loss.item())
        pyplot.plot(loss)
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss Value')
        pyplot.title('Epochs vs. Loss')
        pyplot.savefig('../../result/stage_3_result/loss_plot'+self.method_name+'.png')
        pyplot.clf()

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        accuracy_evaluator = Evaluate_Accuracy(' ', '')
        precision_evaluator = Evaluate_Precision(' ', '')
        recall_evaluator = Evaluate_Recall(' ', '')
        f1_evaluator = Evaluate_F1(' ', '')

        accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        precision_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        recall_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        f1_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}

        print('Accuracy:', accuracy_evaluator.evaluate(), 'Precision:', precision_evaluator.evaluate(),
              'Recall:', recall_evaluator.evaluate(), 'F1:', f1_evaluator.evaluate())
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
