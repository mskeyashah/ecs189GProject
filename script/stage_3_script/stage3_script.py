from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Method_MLP_Changed import Method_MLP_Changed
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('image', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_MLP('multi-layer_perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('Train Test declaration', ' ')

    evaluate_obj = Evaluate_Accuracy('accuracy, precision, recall, F-1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start Original Model ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    """print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish Original Model ************')
    # ------------------------------------------------------

    method_obj = Method_MLP_Changed('multi-layer_perceptron_changed_model', '')

    print('************ Start Changed Model ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Changed Model Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish Changed Model ************') """