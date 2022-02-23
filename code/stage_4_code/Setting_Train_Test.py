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
    counter_obj = collections.Counter()

    def tokenize(self,list):
        final = []
        for loaded_data in list:
            tokens = word_tokenize(loaded_data.decode('utf-8'))
            tokens = [w.lower() for w in tokens]
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            self.counter_obj.update(words)
            final.append(words)
        return final

    def load_run_save_evaluate(self):
        rel_path = self.dataset.dataset_source_folder_path

        #train
        X_train = []
        y_train = []
        #train_pos
        count = 0

        self.dataset.dataset_source_folder_path = self.dataset.dataset_source_folder_path + "/train/pos/"
        directory = os.fsencode(self.dataset.dataset_source_folder_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            self.dataset.dataset_source_file_name = filename
            if count < 10:
                loaded_data = self.dataset.load()
                X_train.append(loaded_data)
                y_train.append(1)
                count = count + 1

        count = 0
        #train_neg
        self.dataset.dataset_source_folder_path = rel_path + "/train/neg/"
        directory = os.fsencode(self.dataset.dataset_source_folder_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            self.dataset.dataset_source_file_name = filename
            if count < 10:
                loaded_data = self.dataset.load()
                X_train.append(loaded_data)
                y_train.append(0)
                count = count +1
        count = 0
        #test_pos
        X_test = []
        y_test = []
        self.dataset.dataset_source_folder_path = rel_path + "/test/pos/"
        directory = os.fsencode(self.dataset.dataset_source_folder_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            self.dataset.dataset_source_file_name = filename
            if count < 10:
                loaded_data = self.dataset.load()
                X_test.append(loaded_data)
                y_test.append(1)
                count = count + 1

        count = 0
        #test_neg
        self.dataset.dataset_source_folder_path = rel_path + "/test/neg/"
        directory = os.fsencode(self.dataset.dataset_source_folder_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            self.dataset.dataset_source_file_name = filename
            if count < 10:
                loaded_data = self.dataset.load()
                X_test.append(loaded_data)
                y_test.append(0)
                count = count + 1

        X_train = self.tokenize(X_train)
        X_test = self.tokenize(X_test)
        result = tt.vocab.vocab(self.counter_obj, min_freq=1)
        #print(result.get_stoi())
        print("done")
        # # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        self.method.vocab = result.get_stoi()
        self.method.n_vocab = len(result.get_stoi())
        learned_result = self.method.run()
        #
        # # save raw ResultModule
        self.result.data = learned_result
        self.result.result_destination_file_name = self.method.method_name+'prediction_result'
        self.result.save()
        #
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        