'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        X_train = []
        y_train = []
        for pair in loaded_data['train']:
            X_train.append(pair['image'])
            y_train.append(pair['label'])

        X_test = []
        y_test = []
        for pair in loaded_data['test']:
            X_test.append(pair['image'])
            y_test.append(pair['label'])


        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.result_destination_file_name = self.method.method_name+'prediction_result'
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        