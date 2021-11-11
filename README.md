# gcn for prediction of protein interactions

利用各种图神经网络进行link prediction of protein interactions。

**Guide**

- [Intro](#Intro)
- [Model](#Model)
- [Dataset](#Dataset)
- [Reference](#reference)

## Intro

目前主要实现基于【data/yeast/yeast.edgelist】下的蛋白质数据进行link prediction。

## Model
### 模型
模型主要使用图神经网络，如gae、vgae等
* 1.GCNModelVAE(src/vgae)：图卷积自编码和变分图卷积自编码(config中可配置使用自编码或变分自编码)， [Variational Graph Auto-Encoders](https://arxiv.org/pdf/1611.07308.pdf) 。

    ![image](https://raw.githubusercontent.com/jiangnanboy/gcn_for_prediction_of_protein_interactions/master/image/vgae.png)
* 2.GCNModelARGA(src/arga)：对抗正则化图自编码，利用gae/vgae作为生成器；一个三层前馈网络作判别器， [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/pdf/1802.04407v2.pdf) 。

    ![image](https://raw.githubusercontent.com/jiangnanboy/gcn_for_prediction_of_protein_interactions/master/image/arga.png)

#### Usage
- 相关参数的配置config见每个模型文件夹中的config.cfg文件，训练和预测时会加载此文件。

- 训练及预测

  ##### 1.GCNModelVAE(src/vgae)
     
     (1).训练
    ```
    from src.vgae.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
    Epoch: 0001 train_loss =  0.73368 val_roc_score =  0.77485 average_precision_score =  0.69364 time= 0.79382
    Epoch: 0002 train_loss =  0.73334 val_roc_score =  0.80637 average_precision_score =  0.74248 time= 0.78920
    Epoch: 0003 train_loss =  0.73341 val_roc_score =  0.85901 average_precision_score =  0.84317 time= 0.78759
    Epoch: 0004 train_loss =  0.73353 val_roc_score =  0.86936 average_precision_score =  0.85909 time= 0.78880
    Epoch: 0005 train_loss =  0.73334 val_roc_score =  0.86945 average_precision_score =  0.86092 time= 0.78438
    Epoch: 0006 train_loss =  0.73353 val_roc_score =  0.87117 average_precision_score =  0.86205 time= 0.78761
    Epoch: 0007 train_loss =  0.73352 val_roc_score =  0.87235 average_precision_score =  0.86407 time= 0.78210
    Epoch: 0008 train_loss =  0.73338 val_roc_score =  0.87317 average_precision_score =  0.86462 time= 0.78477
    Epoch: 0009 train_loss =  0.73341 val_roc_score =  0.87462 average_precision_score =  0.86755 time= 0.78378
    Epoch: 0010 train_loss =  0.73348 val_roc_score =  0.87606 average_precision_score =  0.86853 time= 0.78587
    Epoch: 0011 train_loss =  0.73344 val_roc_score =  0.87686 average_precision_score =  0.86923 time= 0.78406
    Epoch: 0012 train_loss =  0.73331 val_roc_score =  0.87665 average_precision_score =  0.86880 time= 0.78253
    Epoch: 0013 train_loss =  0.73357 val_roc_score =  0.87426 average_precision_score =  0.86521 time= 0.78202
    Epoch: 0014 train_loss =  0.73327 val_roc_score =  0.87218 average_precision_score =  0.86192 time= 0.78299
    Epoch: 0015 train_loss =  0.73336 val_roc_score =  0.87118 average_precision_score =  0.85946 time= 0.78166
    Epoch: 0016 train_loss =  0.73336 val_roc_score =  0.86960 average_precision_score =  0.85835 time= 0.78792
    Epoch: 0017 train_loss =  0.73355 val_roc_score =  0.87126 average_precision_score =  0.85940 time= 0.78401
    Epoch: 0018 train_loss =  0.73357 val_roc_score =  0.87050 average_precision_score =  0.85648 time= 0.78511
    Epoch: 0019 train_loss =  0.73332 val_roc_score =  0.86737 average_precision_score =  0.84906 time= 0.78132
    Epoch: 0020 train_loss =  0.73345 val_roc_score =  0.86632 average_precision_score =  0.84532 time= 0.78603
    
    test roc score: 0.863696753293295
    test ap score: 0.8381410617542567
    ```
      
    (2).预测

    ```
    from src.vgae.predict import Predict
  
    predict = Predict()
    predict.load_model_adj('config_cfg')
    # 会返回原始的图邻接矩阵和经过模型编码后的hidden embedding经过内积解码的邻接矩阵，可以对这两个矩阵进行比对，得出link prediction.
    adj_orig, adj_rec = predict.predict()
    ```
  
  ##### 2.GCNModelARGA(src/arga)
     
     (1).训练
    ```
    from src.arga.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
    Epoch: 0001 train_loss =  2.17176 val_roc_score =  0.77090 average_precision_score =  0.69050 time= 0.81113
    Epoch: 0002 train_loss =  2.16173 val_roc_score =  0.84636 average_precision_score =  0.81340 time= 0.81458
    Epoch: 0003 train_loss =  2.14979 val_roc_score =  0.87660 average_precision_score =  0.86472 time= 0.80898
    Epoch: 0004 train_loss =  2.13698 val_roc_score =  0.87735 average_precision_score =  0.86534 time= 0.80995
    Epoch: 0005 train_loss =  2.12339 val_roc_score =  0.87765 average_precision_score =  0.86592 time= 0.80865
    Epoch: 0006 train_loss =  2.10753 val_roc_score =  0.87756 average_precision_score =  0.86571 time= 0.80748
    Epoch: 0007 train_loss =  2.08996 val_roc_score =  0.87806 average_precision_score =  0.86621 time= 0.80738
    Epoch: 0008 train_loss =  2.06920 val_roc_score =  0.87801 average_precision_score =  0.86623 time= 0.80744
    Epoch: 0009 train_loss =  2.04701 val_roc_score =  0.87795 average_precision_score =  0.86618 time= 0.80932
    Epoch: 0010 train_loss =  2.02241 val_roc_score =  0.87830 average_precision_score =  0.86643 time= 0.80722
    Epoch: 0011 train_loss =  1.99754 val_roc_score =  0.87807 average_precision_score =  0.86620 time= 0.80533
    Epoch: 0012 train_loss =  1.97255 val_roc_score =  0.87749 average_precision_score =  0.86586 time= 0.80859
    Epoch: 0013 train_loss =  1.94664 val_roc_score =  0.87607 average_precision_score =  0.86483 time= 0.80660
    Epoch: 0014 train_loss =  1.92208 val_roc_score =  0.87408 average_precision_score =  0.86320 time= 0.80300
    Epoch: 0015 train_loss =  1.89869 val_roc_score =  0.87290 average_precision_score =  0.86218 time= 0.80400
    Epoch: 0016 train_loss =  1.87584 val_roc_score =  0.87244 average_precision_score =  0.86186 time= 0.80392
    Epoch: 0017 train_loss =  1.85415 val_roc_score =  0.87554 average_precision_score =  0.86400 time= 0.80675
    Epoch: 0018 train_loss =  1.83373 val_roc_score =  0.87653 average_precision_score =  0.86473 time= 0.80762
    Epoch: 0019 train_loss =  1.81515 val_roc_score =  0.87718 average_precision_score =  0.86532 time= 0.80596
    Epoch: 0020 train_loss =  1.79975 val_roc_score =  0.87745 average_precision_score =  0.86551 time= 0.80889
    
    test roc score: 0.8797451083479302
    test ap score: 0.8681038618348471
    ```
      
    (2).预测

    ```
    from src.arga.predict import Predict
  
    predict = Predict()
    predict.load_model_adj('config_cfg')
    # 会返回原始的图邻接矩阵和经过模型编码后的hidden embedding经过内积解码的邻接矩阵，可以对这两个矩阵进行比对，得出link prediction.
    adj_orig, adj_rec = predict.predict()
    ```

## Dataset

   数据来自酵母蛋白质相互作用[yeast](http://snap.stanford.edu/deepnetbio-ismb/ipynb/yeast.edgelist) 。
   数据集的格式如下，具体可见[data](data/yeast/yeast.edgelist)。
   ```
    YLR418C	YOL145C
    YOL145C	YLR418C
    YLR418C	YOR123C
    YOR123C	YLR418C
    ......         ......
   ```
    

## Reference

* [Variational Graph Auto-Encoders](https://arxiv.org/pdf/1611.07308.pdf)
* https://github.com/zfjsail/gae-pytorch/blob/master/gae/utils.py
* https://github.com/tkipf/gae/tree/master/gae
* http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html
* [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/pdf/1802.04407v2.pdf)
* https://github.com/pyg-team/pytorch_geometric
