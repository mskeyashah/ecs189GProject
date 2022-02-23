'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Evaluate_Precision import Evaluate_Precision
from code.stage_4_code.Evaluate_Recall import Evaluate_Recall
from code.stage_4_code.Evaluate_F1 import Evaluate_F1
from matplotlib import pyplot

import torch
from torch import nn
import torch.optim as optim
import numpy as np


class Method_RNN_Text_Classification_Changed(method, nn.Module):
    data = None
    vocab = None
    n_vocab = 0
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.001

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        n_embed = 500
        n_hidden = 400
        n_output = 1  # 1 ("positive") or 0 ("negative")
        n_layers = 2

        self.embedding = nn.Embedding(self.n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        embedded_words = self.embedding(x)  # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)  # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, 400)  # (batch_size*seq_length, n_hidden)
        fc_out = self.fc(lstm_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(len(x), -1)  # (batch_size, seq_length*n_output)

        return sigmoid_out,h

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy(' ', '')
        precision_evaluator = Evaluate_Precision(' ', '')
        recall_evaluator = Evaluate_Recall(' ', '')
        f1_evaluator = Evaluate_F1(' ', '')
        # it will be an iterative gradient updating process


        encoded_reviews = [[self.vocab[word] for word in review] for review in X]
        reviews = []

        for review in encoded_reviews:
            if len(review) >= 500:
                reviews.append(review[:500])
            else:
                reviews.append([0] * (500 - len(review)) + review)

        encoded_reviews = np.array(reviews)
        X = torch.LongTensor(encoded_reviews)
        y = torch.LongTensor(y)

        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred,h = self.forward(X)
            # convert y to torch.tensor as well
            # calculate the training loss

            train_loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%5 == 0:
                accuracy_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                precision_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                recall_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                f1_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}

                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Precision:',
                    precision_evaluator.evaluate(), 'Recall:', recall_evaluator.evaluate(), 'F1:',
                    f1_evaluator.evaluate(), 'Loss:', train_loss.item())


    def test(self, X):
        # do the testing, and result the result
        encoded_reviews = [[self.vocab[word] for word in review] for review in X]
        reviews = []

        for review in encoded_reviews:
            if len(review) >= 500:
                reviews.append(review[:500])
            else:
                reviews.append([0] * (500 - len(review)) + review)

        encoded_reviews = np.array(reviews)
        X = torch.LongTensor(encoded_reviews)
        y_pred,h = self.forward(X)
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
