'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import os
from code.base_class.setting import setting
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import torchtext as tt
import collections



class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):
        adj, features, y_train, y_test, train_mask, test_mask = self.dataset.load_data()
        # # run MethodModule
        self.method.data = {'adj': adj, 'features': features, 'y_train': y_train, 'y_test': y_test,'train_mask': train_mask, 'test_mask':test_mask}

        learned_result = self.method.run()
        #
        # # save raw ResultModule
        self.result.data = learned_result
        self.result.result_destination_file_name = self.method.method_name+'prediction_result'
        self.result.save()
        #
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        