'''
@Author: Cabrite
@Date: 2020-07-02 21:30:39
@LastEditors: Cabrite
@LastEditTime: 2020-07-15 23:21:20
@Description: Do not edit
'''

import numpy as np
import argparse
import utils
import os
import gc

def ParseInputs():
    """对输入进行解包

    Returns:
        class: 解包后的输入
    """
    parser = argparse.ArgumentParser(description='Train mrDAE or Visualize Siamese Weights')
    parser.add_argument("--Mode", type=int, default="0")
    args = parser.parse_args()
    return args

def getGaborFilter():
    """生成Gabor滤波器类

    Returns:
        class: Gabor滤波器类
    """
    ksize = (29, 29)
    Lambda = [1, 2, 3, 4]
    numTheta = 8
    Theta = [np.pi / numTheta * i for i in range(numTheta)]
    Beta = [1]
    Gamma = [0.5, 1]
    RI_Part = 'b'

    #- 获取Gabor滤波器组
    Gabor_Filter = utils.Gabor()
    Gabor_Filter.setParam(ksize, Theta, Lambda, Gamma, Beta, RI_Part)
    return Gabor_Filter

def Load_Data(data_dir, data_name, mode):
    """读取数据集

    Args:
        data_dir (str): 数据集路径
        data_name (str): 数据集名称
        mode (int): 数据集模式 0：MNIST， FashionMNIST   1：SVHN

    Returns:
        array: 数据集
    """
    #* 读取数据集
    return utils.Preprocess_Raw_Data(data_dir, data_name, mode, True, True)

def DataPreprocess(Train_X, Gabor_Filter, ImageBlockSize, numSamples, Whiten=True, saveImages=None, loadImages=None, saveBlocks=None, loadBlocks=None, batchsize=5000):
    """数据集预处理：对图像全图Gabor滤波 -> 随机采样指定大小图像 -> PCA白化

    Args:
        Train_X (array): 训练集
        Gabor_Filter (class): Gabor滤波器
        ImageBlockSize (tuple): 采样快大小
        numSamples (int): 采样数量
        Whiten (bool, optional): 是否白化. Defaults to True.
        saveImages (str, optional): Gabor图像保存路径. Defaults to None.
        loadImages (str, optional): Gabor图像读取路径. Defaults to None.
        saveBlocks ([str, str], optional): 采样图像及其Gabor图像保存路径. Defaults to None.
        loadBlocks ([str, str], optional): 采样图像及其Gabor图像读取路径. Defaults to None.
        batchsize (int, optional): 批处理大小. Defaults to 5000.

    Returns:
        array: 图像块，图像块对应Gabor，白化均值，白化矩阵
    """
    #* Gabor图像
    if loadImages:
        Train_Gabor = utils.LoadGaborImages(loadImages)
    else:
        Train_Gabor = utils.GaborAllImages(Gabor_Filter, Train_X, batchsize=batchsize, isSavingData=saveImages)

    #* 图像采样
    if loadBlocks:
        Image_Blocks, Image_Blocks_Gabor = utils.LoadRandomImageBlocks(loadBlocks)
    else:
        Image_Blocks, Image_Blocks_Gabor = utils.RandomSamplingImageBlocks(Train_X, Train_Gabor, Gabor_Filter, ImageBlockSize, numSamples, isSavingData=saveBlocks)

    Image_Blocks, Whiten_Average, Whiten_U = utils.PCA_Whiten(Image_Blocks, Whiten)

    if Whiten:
        if not os.path.exists('./Model_mrDAE'):
            os.mkdir('./Model_mrDAE')
        np.save('./Model_mrDAE/Whiten_Average.npy', Whiten_Average)
        np.save('./Model_mrDAE/Whiten_MatrixU.npy', Whiten_U)

    del Train_Gabor
    gc.collect()

    return Image_Blocks, Image_Blocks_Gabor, Whiten_Average, Whiten_U


#- 训练网络
def Build_Networks():
    """训练网络
    """
    #* 数据保存路径
    # Data_Preserve_Dir = './Data_Preprocessed/Gabored_Train.npy'
    # Image_Blocks_Dir = ['./Data_Preprocessed/ImageBlocks.npy', './Data_Preprocessed/ImageBlocksGabor.npy']
    
    #- 初始化及预处理
    isWhiten = True
    #* Gabor 滤波器
    Gabor_Filter = getGaborFilter()
    #* 载入数据
    Train_X, Train_Y, Test_X, Test_Y = Load_Data("./Datasets/MNIST_Data", "MNIST")
    #* 截取图像块、数据预处理
    Image_Blocks, Image_Blocks_Gabor, Whiten_Average, Whiten_U = DataPreprocess(Train_X, Gabor_Filter, (11, 11), 400000, isWhiten)
    
    #- mrDAE
    #* 初始化mrDAE参数
    mrDAE = utils.MultiResolutionDAE()
    mrDAE.set_Gabor_Filter(Gabor_Filter)
    mrDAE.set_AE_Input_Data(Image_Blocks, Image_Blocks_Gabor)
    mrDAE.set_AE_Parameters(n_Hiddens=1024, reconstruction_reg=0.5, measurement_reg=0.1, sparse_reg=0.1, gaussian=0.02, batch_size=500, display_step=1)
    mrDAE.set_TiedAE_Training_Parameters(epochs=650, lr_init=2e-1, lr_decay_step=4, lr_decay_rate=0.98)

    #* 训练mrDAE
    mrDAE.Build_TiedAutoEncoderNetwork()

    #* 删除mrDAE输入数据，释放内存
    del Image_Blocks, Image_Blocks_Gabor
    gc.collect()

    #* 提取mrDAE特征
    Train_feature, Test_feature = mrDAE.get_mrDAE_Train_Test_Feature(Train_X, Test_X, isWhiten, Whiten_Average, Whiten_U)

    #* 删除mrDAE，释放内存
    del mrDAE
    gc.collect()


    #- 分类器
    #* 初始化mlp参数
    mlp = utils.mrDAE_Classifier()
    mlp.set_MLP_Training_Parameters(n_Hiddens=2048, epochs=500, batchsize=200, gaussian=0.02, lr_init=2e-1, lr_step=4, lr_rate=0.98, display_step=1, dropout=1)
    mlp.set_MLP_Input_Data(Train_feature, Train_Y, Test_feature, Test_Y)
    
    #* 训练mlp
    mlp.Build_ClassificationNetwork()


#- 可视化
def VisualizeSiamese():
    """网络可视化
    """
    #* Gabor 滤波器
    Gabor_Filter = getGaborFilter()

    #* 初始化mrDAE参数
    mrDAE = utils.MultiResolutionDAE()
    mrDAE.set_Gabor_Filter(Gabor_Filter)

    #* 可视化
    mrDAE.Visualization()


if __name__ == "__main__":
    args = ParseInputs()

    if args.Mode == 0:
        Build_Networks()
    else:
        VisualizeSiamese()
    

    
