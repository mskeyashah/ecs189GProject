from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN_Text_Classification import Method_RNN_Text_Classification

from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test import Setting_Train_Test_Split
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('text classif', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
    #data_obj.dataset_source_file_name = 'MNIST'
    method_obj = Method_RNN_Text_Classification('Text classif', '')
    #
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/Text_Classif_'
    result_obj.result_destination_file_name = 'prediction_result'
    #
    setting_obj = Setting_Train_Test_Split('Train Test declaration', ' ')
    #
    evaluate_obj = Evaluate_Accuracy('accuracy, precision, recall, F-1', '')
    # # ------------------------------------------------------
    #
    # #---- running section ---------------------------------
    # print('************ Start MNIST Model ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish MNIST Model ************')
    # # ------------------------------------------------------
