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
* 1.GCNModelVAE(src/vgae)：图卷积自编码和变分图卷积自编码(config中可配置使用自编码或变分自编码)，利用gae/vgae作为编码器，InnerProductDecoder作解码器。 [Variational Graph Auto-Encoders](https://arxiv.org/pdf/1611.07308.pdf) 。

    ![image](https://raw.githubusercontent.com/jiangnanboy/gcn_for_prediction_of_protein_interactions/master/image/vgae.png)
* 2.GCNModelARGA(src/arga)：对抗正则化图自编码，利用gae/vgae作为生成器；一个三层前馈网络作判别器。 [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/pdf/1802.04407v2.pdf) 。

    ![image](https://raw.githubusercontent.com/jiangnanboy/gcn_for_prediction_of_protein_interactions/master/image/arga.png)
* 3.GATModelVAE(src/graph_att_gae)：基于图注意力的图卷积自编码和变分图卷积自编码(config中可配置使用自编码或变分自编码)，利用gae/vgae作为编码器，InnerProductDecoder作解码器。这是我在以上【1】方法的基础上加入了一层图注意力层，关于图注意力可见【Reference】中的【GRAPH ATTENTION NETWORKS】。
* 4.GATModelGAN(src/graph_att_gan)：基于图注意力的对抗正则化图自编码，利用gae/vgae作为生成器；一个三层前馈网络作判别器，这是我在以上【2】方法的基础上加入了一层图注意力层，关于图注意力可见【Reference】中的【GRAPH ATTENTION NETWORKS】。

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
  
  ##### 3.GATModelVAE(src/graph_att_gae)
    (1).训练
    ```
    from src.graph_att_gae.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
    ```
    Epoch: 0001 train_loss =  0.73325 val_roc_score =  0.72741 average_precision_score =  0.62188 time= 0.80873
    Epoch: 0002 train_loss =  0.73332 val_roc_score =  0.86092 average_precision_score =  0.84305 time= 0.80357
    Epoch: 0003 train_loss =  0.73340 val_roc_score =  0.87373 average_precision_score =  0.86307 time= 0.79868
    Epoch: 0004 train_loss =  0.73317 val_roc_score =  0.87535 average_precision_score =  0.86470 time= 0.80704
    Epoch: 0005 train_loss =  0.73351 val_roc_score =  0.87536 average_precision_score =  0.86470 time= 0.80163
    Epoch: 0006 train_loss =  0.73311 val_roc_score =  0.87543 average_precision_score =  0.86479 time= 0.81234
    Epoch: 0007 train_loss =  0.73357 val_roc_score =  0.87559 average_precision_score =  0.86492 time= 0.80755
    Epoch: 0008 train_loss =  0.73312 val_roc_score =  0.87489 average_precision_score =  0.86442 time= 0.81295
    Epoch: 0009 train_loss =  0.73326 val_roc_score =  0.87479 average_precision_score =  0.86425 time= 0.81095
    Epoch: 0010 train_loss =  0.73342 val_roc_score =  0.87409 average_precision_score =  0.86374 time= 0.81146
    Epoch: 0011 train_loss =  0.73347 val_roc_score =  0.87286 average_precision_score =  0.86265 time= 0.80884
    Epoch: 0012 train_loss =  0.73341 val_roc_score =  0.87209 average_precision_score =  0.86150 time= 0.81265
    Epoch: 0013 train_loss =  0.73355 val_roc_score =  0.87135 average_precision_score =  0.86060 time= 0.80919
    Epoch: 0014 train_loss =  0.73330 val_roc_score =  0.87004 average_precision_score =  0.85823 time= 0.81394
    Epoch: 0015 train_loss =  0.73322 val_roc_score =  0.86950 average_precision_score =  0.85719 time= 0.80923
    Epoch: 0016 train_loss =  0.73338 val_roc_score =  0.86871 average_precision_score =  0.85600 time= 0.81212
    Epoch: 0017 train_loss =  0.73346 val_roc_score =  0.86896 average_precision_score =  0.85637 time= 0.81340
    Epoch: 0018 train_loss =  0.73346 val_roc_score =  0.86937 average_precision_score =  0.85592 time= 0.81007
    Epoch: 0019 train_loss =  0.73339 val_roc_score =  0.87089 average_precision_score =  0.85818 time= 0.79646
    Epoch: 0020 train_loss =  0.73311 val_roc_score =  0.87144 average_precision_score =  0.85820 time= 0.80058
    
    test roc score: 0.8714519347032031
    test ap score: 0.8593487257170946
    ```
      
    (2).预测

    ```
    from src.graph_att_gae.predict import Predict
  
    predict = Predict()
    predict.load_model_adj('config_cfg')
    # 会返回原始的图邻接矩阵和经过模型编码后的hidden embedding经过内积解码的邻接矩阵，可以对这两个矩阵进行比对，得出link prediction.
    adj_orig, adj_rec = predict.predict()
    ```
  
  ##### 4.GATModelGAN(src/graph_att_gan)
    (1).训练
    ```
    from src.graph_att_gan.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
    ```
    Epoch: 0001 train_loss =  2.19019 val_roc_score =  0.72951 average_precision_score =  0.62579 time= 0.82239
    Epoch: 0002 train_loss =  2.18480 val_roc_score =  0.85762 average_precision_score =  0.84009 time= 0.80851
    Epoch: 0003 train_loss =  2.17933 val_roc_score =  0.87056 average_precision_score =  0.86129 time= 0.80706
    Epoch: 0004 train_loss =  2.17487 val_roc_score =  0.87241 average_precision_score =  0.86276 time= 0.80689
    Epoch: 0005 train_loss =  2.16849 val_roc_score =  0.87377 average_precision_score =  0.86433 time= 0.82771
    Epoch: 0006 train_loss =  2.16361 val_roc_score =  0.87430 average_precision_score =  0.86472 time= 0.80936
    Epoch: 0007 train_loss =  2.15645 val_roc_score =  0.87491 average_precision_score =  0.86516 time= 0.80772
    Epoch: 0008 train_loss =  2.15187 val_roc_score =  0.87483 average_precision_score =  0.86499 time= 0.80491
    Epoch: 0009 train_loss =  2.14528 val_roc_score =  0.87543 average_precision_score =  0.86555 time= 0.80434
    Epoch: 0010 train_loss =  2.13976 val_roc_score =  0.87557 average_precision_score =  0.86562 time= 0.81330
    Epoch: 0011 train_loss =  2.13290 val_roc_score =  0.87565 average_precision_score =  0.86563 time= 0.80601
    Epoch: 0012 train_loss =  2.12783 val_roc_score =  0.87552 average_precision_score =  0.86552 time= 0.80824
    Epoch: 0013 train_loss =  2.12035 val_roc_score =  0.87602 average_precision_score =  0.86610 time= 0.80804
    Epoch: 0014 train_loss =  2.11536 val_roc_score =  0.87639 average_precision_score =  0.86620 time= 0.80723
    Epoch: 0015 train_loss =  2.11036 val_roc_score =  0.87596 average_precision_score =  0.86608 time= 0.80584
    Epoch: 0016 train_loss =  2.10575 val_roc_score =  0.87614 average_precision_score =  0.86601 time= 0.80662
    Epoch: 0017 train_loss =  2.10216 val_roc_score =  0.87622 average_precision_score =  0.86612 time= 0.80530
    Epoch: 0018 train_loss =  2.09839 val_roc_score =  0.87625 average_precision_score =  0.86605 time= 0.81081
    Epoch: 0019 train_loss =  2.09667 val_roc_score =  0.87594 average_precision_score =  0.86590 time= 0.80764
    Epoch: 0020 train_loss =  2.09316 val_roc_score =  0.87595 average_precision_score =  0.86593 time= 0.80807
    
    test roc score: 0.8769472359794324
    test ap score: 0.8697789753279398
    ```
      
    (2).预测

    ```
    from src.graph_att_gan.predict import Predict
  
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
* [GRAPH ATTENTION NETWORKS](https://arxiv.org/pdf/1710.10903.pdf)
