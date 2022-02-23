'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from torch.utils.data import DataLoader
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


class Method_RNN_Text_Generation_Changed(method, nn.Module):
    data = None
    n_vocab = 0
    # it defines the max rounds to train the model
    max_epoch = 25
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.01

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.lstm_size = 64
        self.embedding_dim = 64
        self.num_layers = 4

        n_vocab = 6925
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

    def forward(self, x, prev_state):
        '''Forward propagation'''

        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy(' ', '')
        precision_evaluator = Evaluate_Precision(' ', '')
        recall_evaluator = Evaluate_Recall(' ', '')
        f1_evaluator = Evaluate_F1(' ', '')
        losslist = []
        # it will be an iterative gradient updating process

        dataloader = DataLoader(self.data, batch_size=500)
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            state_h, state_c = self.init_state(4)
            for batch,(x,y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))
                loss = loss_function(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()
                if batch % 5 == 0:
                    accuracy_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                    precision_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                    recall_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                    f1_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}

                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Precision:',
                          precision_evaluator.evaluate(), 'Recall:', recall_evaluator.evaluate(), 'F1:',
                          f1_evaluator.evaluate(), 'Loss:', loss.item())
                losslist.append(loss.item())
        pyplot.plot(losslist)
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss Value')
        pyplot.title('Epochs vs. Loss')
        pyplot.savefig('../../result/stage_4_result/loss_plot'+self.method_name+'.png')
        pyplot.clf()

    def test(self, X):

        words = X.split(' ')
        state_h, state_c = self.init_state(len(words))

        for i in range(0, 50):
            x = torch.tensor([[self.data.word_to_index[w] for w in words[i:]]])
            y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.data.index_to_word[word_index])

        return words

    def run(self):
        print('--start training...')
        self.train()
        print('--start testing...')
        pred_y = self.test("Knock knock. Who's")
        print(pred_y)
        pred_y = self.test("What do you")
        print(pred_y)
        pred_y = self.test("What does a")
        print(pred_y)
        pred_y = self.test("What happens to")
        print(pred_y)