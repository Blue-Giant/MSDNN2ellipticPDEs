"""
@author: LXA
 Data: 2020 年 5 月 31 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import DNN_data
import time
import CPDNN_base
import DNN_base
import DNN_tools
import pLaplace_eqs1d
import laplace_eqs1d
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Laplace type for problem: %s\n' % (R_dic['laplace_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)

    if R['laplace_type'] == 'p_laplace':
        DNN_tools.log_string('order of p-Laplacian: %s\n' % (R_dic['order2laplace']), log_fileout)
        DNN_tools.log_string('epsilon: %f\n' % (R_dic['epsilon']), log_fileout)

    if R_dic['variational_loss'] == 1:
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


def solve_laplace(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['boundary_penalty']                # Regularization parameter for boundary conditions
    wb_regular = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    act_func = R['activate_func']

    # ------- set the problem ---------
    input_dim = R['input_dim']
    out_dim = R['output_dim']

    region_l = 0.0
    region_r = 1.0
    if R['laplace_type'] == 'general_laplace':
        # -laplace u = f
        region_l = 0.0
        region_r = 1.0
        f, u_true, u_left, u_right = laplace_eqs1d.get_laplace_infos(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_l, right_top=region_r, laplace_name=R['equa_name'])
    elif R['laplace_type'] == 'p_laplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
        p_index = R['order2laplace']
        epsilon = R['epsilon']
        # index2freq = R['flag2freq'] #一直报错，为什么？
        index2freq = [1, 1.0 / epsilon]
        if 2 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_2laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        elif 3 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_3laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        elif 5 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_5laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        elif 8 == p_index:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_8laplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon)
        else:
            region_l = 0.0
            region_r = 1.0
            u_true, f, A_eps, u_left, u_right = pLaplace_eqs1d.get_infos_pLaplace(
                in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, p=p_index, eps=epsilon,
                eqs_name=R['eqs_name'])

    if R['model'] == 'PDE_subDNNs_scale' or R['model'] == 'PDE_subDNNs_adapt_scale':
        W_lists = []
        B_lists = []
        nums2nets = R['num2subnet']
        for isubnes in range(nums2nets):
            flag1 = 'WB' + str(isubnes) + 'subnet'
            Weights, Biases = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag1)
            W_lists.append(Weights)
            B_lists.append(Biases)
    else:
        flag1 = 'WB'
        Weights, Biases = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag1)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X_it = tf.placeholder(tf.float32, name='X_it', shape=[None, input_dim])                # * 行 1 列
            X_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])      # * 行 1 列
            X_right_bd = tf.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])    # * 行 1 列
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            # 供选择的网络模式
            if R['model'] == 'PDE_DNN':
                # 在变量的内部区域训练
                U_NN = DNN_base.PDE_DNN(X_it, Weights, Biases, hidden_layers, activate_name=act_func)
                # 左右边界训练，使得训练结果(参数和偏置模型)逼近边界
                ULeft_NN = DNN_base.PDE_DNN(X_left_bd, Weights, Biases, hidden_layers, activate_name=act_func)
                URight_NN = DNN_base.PDE_DNN(X_right_bd, Weights, Biases, hidden_layers, activate_name=act_func)
            elif R['model'] == 'PDE_DNN_BN':
                U_NN = DNN_base.PDE_DNN_BN(X_it, Weights, Biases, hidden_layers, activate_name=act_func,
                                           is_training=train_opt)
                ULeft_NN = DNN_base.PDE_DNN_BN(X_left_bd, Weights, Biases, hidden_layers, activate_name=act_func,
                                               is_training=train_opt)
                URight_NN = DNN_base.PDE_DNN_BN(X_right_bd, Weights, Biases, hidden_layers, activate_name=act_func,
                                                is_training=train_opt)
            elif R['model'] == 'PDE_DNN_scale':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN = DNN_base.PDE_DNN_scale(X_it, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                ULeft_NN = DNN_base.PDE_DNN_scale(X_left_bd, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                URight_NN = DNN_base.PDE_DNN_scale(X_right_bd, Weights, Biases, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'PDE_subDNNs_scale':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN = DNN_base.PDE_subDNNs_scale(X_it, W_lists, B_lists, hidden_layers, freq, activate_name=act_func)
                ULeft_NN = DNN_base.PDE_subDNNs_scale(X_left_bd, W_lists, B_lists, hidden_layers, freq, activate_name=act_func)
                URight_NN = DNN_base.PDE_subDNNs_scale(X_right_bd, W_lists, B_lists, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'PDE_DNN_adapt_scale':
                freq_flags = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                freq = freq_flags*0.005
                U_NN = DNN_base.PDE_DNN_adapt_scale(X_it, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                ULeft_NN = DNN_base.PDE_DNN_adapt_scale(X_left_bd, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                URight_NN = DNN_base.PDE_DNN_adapt_scale(X_right_bd, Weights, Biases, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'PDE_subDNNs_adapt_scale':
                freq_flags = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                freq = freq_flags*0.005
                U_NN = DNN_base.PDE_DNN_adapt_scale(X_it, W_lists, B_lists, hidden_layers, freq, activate_name=act_func)
                ULeft_NN = DNN_base.PDE_DNN_adapt_scale(X_left_bd, W_lists, B_lists, hidden_layers, freq, activate_name=act_func)
                URight_NN = DNN_base.PDE_DNN_adapt_scale(X_right_bd, W_lists, B_lists, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'PDE_DNN_FourierBase':
                freqs = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN = DNN_base.PDE_DNN_FourierBase(X_it, Weights, Biases, hidden_layers, freqs, activate_name=act_func)
                ULeft_NN = DNN_base.PDE_DNN_FourierBase(X_left_bd, Weights, Biases, hidden_layers, freqs, activate_name=act_func)
                URight_NN = DNN_base.PDE_DNN_FourierBase(X_right_bd, Weights, Biases, hidden_layers, freqs, activate_name=act_func)

            # 变分形式的loss of interior，训练得到的 U_NN 是 * 行 1 列, 因为 一个点对(x,y) 得到一个 u 值
            if R['variational_loss'] == 1:
                dU_NN = tf.gradients(U_NN, X_it)      # * 行 1 列
                if R['laplace_type'] == 'general_laplace':
                    laplace_norm = tf.reduce_sum(tf.square(dU_NN), axis=-1)
                    loss_it_variational = (1.0 / 2) * tf.reshape(laplace_norm, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)
                elif R['laplace_type'] == 'p_laplace':
                    # a_eps = A_eps(X_it)                          # * 行 1 列
                    a_eps = 1 / (2 + tf.cos(2 * np.pi * X_it / epsilon))
                    laplace_p_pow = tf.reduce_sum(a_eps*tf.pow(tf.abs(dU_NN), p_index), axis=-1)
                    loss_it_variational = (1.0 / p_index) * tf.reshape(laplace_p_pow, shape=[-1, 1]) - \
                                           tf.multiply(tf.reshape(f(X_it), shape=[-1, 1]), U_NN)

                loss_it = tf.reduce_mean(loss_it_variational)*(region_r-region_l)

            U_left = u_left(X_left_bd)
            U_right = u_right(X_right_bd)
            loss_bd_square = tf.square(ULeft_NN - U_left) + tf.square(URight_NN - U_right)
            loss_bd = tf.reduce_mean(loss_bd_square)

            if R['regular_weight_model'] == 'L1':
                regular_WB = DNN_base.regular_weights_biases_L1(Weights, Biases)    # 正则化权重和偏置 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB = DNN_base.regular_weights_biases_L2(Weights, Biases)    # 正则化权重和偏置 L2正则化
            else:
                regular_WB = tf.constant(0.0)                                         # 无正则化权重参数

            penalty_WB = wb_regular * regular_WB
            loss = loss_it + bd_penalty * loss_bd + wb_regular * regular_WB       # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            U_true = u_true(X_it)
            mean_square_error = tf.reduce_mean(tf.square(U_true - U_NN))
            residual_error = mean_square_error / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    testing_epoch = []

    test_batch_size = 1000
    test_x_bach = np.reshape(np.linspace(region_l, region_r, num=test_batch_size), [-1, 1])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            x_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r)
            xl_bd_batch, xr_bd_batch = DNN_data.rand_bd_1D(batchsize_bd, input_dim, region_a=region_l, region_b=region_r)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            _, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp, train_res_tmp, p_WB = sess.run(
                [train_my_loss, loss_it, loss_bd, loss, mean_square_error, residual_error, penalty_WB],
                feed_dict={X_it: x_it_batch, X_left_bd: xl_bd_batch, X_right_bd: xr_bd_batch,
                           in_learning_rate: tmp_lr, bd_penalty: temp_penalty_bd})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)
            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, p_WB, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp,
                    train_res_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                testing_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, u_predict2test = sess.run([U_true, U_NN], feed_dict={X_it: test_x_bach, train_opt: train_option})
                mse2test = np.mean(np.square(u_true2test - u_predict2test))
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)

                DNN_tools.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)

        # -----------------------  save training results to mat files, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ----------------------  save testing results to mat files, then plot them --------------------------------
        saveData.save_2testSolus2mat(u_true2test, u_predict2test, actName='utrue', actName1=act_func, outPath=R['FolderName'])
        plotData.plot_2solutions2test(u_true2test, u_predict2test, coord_points2test=test_x_bach,
                                      batch_size2test=test_batch_size, seedNo=R['seed'], outPath=R['FolderName'],
                                      subfig_type=R['subfig_type'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, testing_epoch, actName=act_func,
                                  seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    store_file = 'laplace1d'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # R['laplace_type'] = 'general_laplace'
    # R['eqs_name'] = 'PDE1'
    # R['eqs_name'] = 'PDE2'
    # R['eqs_name'] = 'PDE3'
    # R['eqs_name'] = 'PDE4'
    # R['eqs_name'] = 'PDE5'
    # R['eqs_name'] = 'PDE6'
    # R['eqs_name'] = 'PDE7'

    R['laplace_type'] = 'p_laplace'
    R['eqs_name'] = 'multi_scale'

    if R['laplace_type'] == 'general_laplace':
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
    elif R['laplace_type'] == 'p_laplace':
        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)              # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2laplace'] = order

    R['input_dim'] = 1                         # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                        # 输出维数
    R['variational_loss'] = 1                  # PDE变分

    # ---------------------------- Setup of DNN -------------------------------
    R['batch_size2interior'] = 3000            # 内部训练数据的批大小
    R['batch_size2boundary'] = 500             # 边界训练数据大小

    R['weight_biases_model'] = 'general_model'
    # R['weight_biases_model'] = 'phase_shift_model'

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight_biases'] = 0.000     # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025  # Regularization parameter for weights

    R['activate_stage_penalty'] = 1
    R['boundary_penalty'] = 100                           # Regularization parameter for boundary conditions

    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器

    R['hidden_layers'] = (30, 20, 20, 15, 15, 10)
    # R['hidden_layers'] = (80, 60, 60, 40, 20, 20)
    # R['hidden_layers'] = (100, 80, 60, 60, 40, 40, 20)
    # R['hidden_layers'] = (200, 100, 80, 50, 30)
    # R['hidden_layers'] = (300, 200, 150, 150, 100, 50, 50)
    # R['hidden_layers'] = (400, 300, 300, 200, 100, 100, 50)
    # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
    # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)
    # R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100)
    # R['hidden_layers'] = (500, 300, 200, 200, 100, 100, 50)
    # R['hidden_layers'] = (1000, 800, 600, 400, 200)
    # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)
    # R['hidden_layers'] = (2000, 1500, 1000, 500, 250)

    # R['model'] = 'PDE_DNN'                           # 使用的网络模型
    # R['model'] = 'PDE_DNN_BN'
    # R['model'] = 'PDE_DNN_scale'
    # R['model'] = 'PDE_DNN_adapt_scale'
    # R['model'] = 'PDE_DNN_FourierBase'
    R['model'] = 'PDE_subDNNs_scale'
    # R['model'] = 'PDE_subDNNs_adapt_scale'

    # R['activate_func'] = 'relu'
    R['activate_func'] = 'singauss'
    # R['activate_func'] = 'tanh'
    # R['activate_func']' = leaky_relu'
    # R['activate_func'] = 'srelu'
    # R['activate_func'] = 's2relu'
    # R['activate_func'] = 'leaklysrelu'
    # R['activate_func'] = 'slrelu'
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'selu'
    # R['activate_func'] = 'phi'

    R['plot_ongoing'] = 0
    R['subfig_type'] = 1
    R['num2subnet'] = 4

    solve_laplace(R)

