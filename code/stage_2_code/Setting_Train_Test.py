'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.base_class.setting import setting
import numpy as np

class Setting_Train_Test_Split(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        data_obj = Dataset_Loader('test.csv', '')
        data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
        data_obj.dataset_source_file_name = 'test.csv'
        data1 = data_obj.load()
        X_test = data1['X']
        y_test = data1['y']
        X_train = loaded_data['X']
        y_train = loaded_data['y']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        