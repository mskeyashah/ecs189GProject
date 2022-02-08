from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from code.stage_3_code.Method_CNN_MNIST_Changed import Method_CNN_MNIST_Changed
from code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from code.stage_3_code.Method_CNN_CIFAR_Changed import Method_CNN_CIFAR_Changed

from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test import Setting_Train_Test_Split
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
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
    data_obj.dataset_source_file_name = 'MNIST'
    #28x28
    method_obj = Method_CNN_MNIST('MNIST', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/MNIST_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('Train Test declaration', ' ')

    evaluate_obj = Evaluate_Accuracy('accuracy, precision, recall, F-1', '')
    # ------------------------------------------------------

    #---- running section ---------------------------------
    #print('************ Start MNIST Model ************')
    #setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    #setting_obj.print_setup_summary()
    #mean_score, std_score = setting_obj.load_run_save_evaluate()
    #print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish MNIST Model ************')
    # ------------------------------------------------------

    method_obj = Method_CNN_MNIST_Changed('MNIST_Changed', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/MNIST_Changed_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('Train Test declaration', ' ')

    evaluate_obj = Evaluate_Accuracy('accuracy, precision, recall, F-1', '')
    # ------------------------------------------------------

    #---- running section ---------------------------------
    print('************ Start Changed MNIST Model ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish Changed MNIST Model ************')



    # data_obj.dataset_source_file_name = 'CIFAR'
    # # #32x32x3
    # method_obj = Method_CNN_CIFAR('CIFAR', '')
    # #
    # result_obj.result_destination_folder_path = '../../result/stage_3_result/CIFAR_'
    # #
    # # # ---- running section ---------------------------------
    # print('************ Start CIFAR Model ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish CIFAR Model ************')
    #
    # # #32x32x3
    # method_obj = Method_CNN_CIFAR_Changed('CIFAR_Changed', '')
    # #
    # result_obj.result_destination_folder_path = '../../result/stage_3_result/CIFAR_Changed_'
    # #
    # # # ---- running section ---------------------------------
    # print('************ Start Changed CIFAR Model ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish Changed CIFAR Model ************')
    #
    #
    # #------------------------------------------------------

    #
    #
    method_obj = Method_CNN_ORL('ORL', '')
    data_obj.dataset_source_file_name = 'ORL'
    #112x92x3

    result_obj.result_destination_folder_path = '../../result/stage_3_result/ORL_'

    # ---- running section ---------------------------------
    print('************ Start ORL Model ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    #mean_score, std_score = setting_obj.load_run_save_evaluate()
    """print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish CIFAR Model ************')
    # ------------------------------------------------------ """