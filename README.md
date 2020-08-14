# Multi-Resolution-DAE
+ This is the prototype of Multi-Resolution Denoising Auto-Encoder.

## Input Parameters
+ python mrDAE.py
  1. `--Mode` , `-m` ：Select Run Mode. Default : `0`
     1. 0 : Train mrDAE, Extract Feature, Train Classification.
     2. 1 : Load Trained mrDAE, Extract Feature, Train Classification.
     3. -1 : Visualize the weights of Siamese part.
  2. `--DefaultPath` , `-p` ：Select Dataset Parent Directory. Default : `"./Datasets"`
  3. `--Dataset` , `-d` ：Select Dataset to Train & Test. Default : `MNIST`
  4. `--TFInfo` , `-i` ： Choose TensorFlow Information Level. Dafaykt : `0`
     1. All Messages Displayed. 1: 
     2. No Info
     3. No Info and Warnings
     4. No Info, Warnings and Errors
  5. `--GPUs` , `-g` ：Select GPUs to use. Default : `None`
  6. `--sync` , `--no-sync` ：Select Whether Synchronize Gabor & Sampling Procedure. 
     + **Sync can help reduce the usage of memory while increasing the consumption of time**

+ Training:
  + python mrDAE.py --Mode=0 --Dataset='MNIST' --sync
  + python mrDAE.py --Mode=0 --Dataset='Fashion_MNIST' --sync
  + python mrDAE.py --Mode=0 --Dataset='SVHN' --sync
  + python mrDAE.py --Mode=0 --Dataset='CIFAR10' --sync