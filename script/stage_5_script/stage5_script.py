from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GCN_Cora import Method_GCN_Cora
from code.stage_5_code.Method_GCN_Cora_Changed import Method_GCN_Cora_Changed
from code.stage_5_code.Method_GCN_Citeseer import Method_GCN_Citeseer
from code.stage_5_code.Method_GCN_Citeseer_Changed import Method_GCN_Citeseer_Changed
from code.stage_5_code.Method_GCN_Pubmed import Method_GCN_Pubmed
from code.stage_5_code.Method_GCN_Pubmed_Changed import Method_GCN_Pubmed_Changed
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting_Train_Test import Setting_Train_Test_Split
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('GCN Cora', '')
    method_obj = Method_GCN_Cora('GCN Cora', '')
    #
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_Cora'
    result_obj.result_destination_file_name = 'prediction_result'
    #
    setting_obj = Setting_Train_Test_Split('Train Test declaration', ' ')
    #
    evaluate_obj = Evaluate_Accuracy('accuracy, precision, recall, F-1', '')
    # # ------------------------------------------------------
    #
    #---- running section ---------------------------------
    # print('************ Start Cora GCN Model ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate("cora")
    # print('************ Overall Performance ************')
    # print('Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish Cora GCN Model ************')
    # # ------------------------------------------------------
    # #
    # method_obj = Method_GCN_Cora_Changed('GCN Cora Changed', '')
    # result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_Cora_Changed'
    # # ---- running section ---------------------------------
    # print('************ Start Changed  Cora GCN Model ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate("cora")
    # print('************ Overall Performance ************')
    # print('Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish Changed Cora GCN Model ************')
    # # ------------------------------------------------------

    data_obj = Dataset_Loader('GCN Citeseer', '')
    method_obj = Method_GCN_Citeseer('GCN Citeseer', '')

    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_Citeseer'
    # ---- running section ---------------------------------
    # print('************ Start Citeseer GCN Model ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate("citeseer")
    # print('************ Overall Performance ************')
    # print('Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish Citeseer GCN Model ************')
    # # ------------------------------------------------------
    #
    # method_obj = Method_GCN_Citeseer_Changed('GCN Citeseer Changed', '')
    #
    # result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_Citeseer_Changed'
    # # ---- running section ---------------------------------
    # print('************ Start Changed Citeseer GCN Model ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate("citeseer")
    # print('************ Overall Performance ************')
    # print('Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish Changed Citeseer GCN Model ************')
    # # ------------------------------------------------------

    data_obj = Dataset_Loader('GCN Pubmed', '')
    method_obj = Method_GCN_Pubmed('GCN Pubmed', '')

    # result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_Pubmed'
    # # ---- running section ---------------------------------
    # print('************ Start Pubmed GCN Model ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate("Pubmed")
    # print('************ Overall Performance ************')
    # print('Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish Pubmed GCN Model ************')
    # # ------------------------------------------------------

    method_obj = Method_GCN_Pubmed_Changed('GCN Pubmed Changed', '')

    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_Pubmed_Changed'
    # ---- running section ---------------------------------
    print('************ Start Changed Pubmed GCN Model ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate("Pubmed")
    print('************ Overall Performance ************')
    print('Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish Changed Pubmed GCN Model ************')
    # ------------------------------------------------------





