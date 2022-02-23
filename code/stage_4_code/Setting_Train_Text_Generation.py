'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting



class Setting_Train_Text_Generation(setting):

    def load_run_save_evaluate(self):
        self.method.data = self.dataset
        self.method.run()
        
        return None, None

        