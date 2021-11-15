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
* 5.NHGATModelVAE(src/graph_nheads_att_gae)：基于图多头注意力的图卷积自编码和变分图卷积自编码(config中可配置使用自编码或变分自编码)，利用gae/vgae作为编码器，InnerProductDecoder作解码器。此方法是在【3】方法的基础上将图注意力层改为多头注意力层。
* 6.NHGATModelGAN(src/graph_nheads_att_gan)：基于图多头注意力的对抗正则化图自编码，利用gae/vgae作为生成器；一个三层前馈网络作判别器，此方法在【4】方法的基础上将图注意力层改为多头注意力层。

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
    Epoch: 0001 train_loss =  1.84734 val_roc_score =  0.76573 average_precision_score =  0.68083 time= 0.80005
    Epoch: 0002 train_loss =  1.83824 val_roc_score =  0.87289 average_precision_score =  0.86317 time= 0.80361
    Epoch: 0003 train_loss =  1.80761 val_roc_score =  0.87641 average_precision_score =  0.86590 time= 0.80121
    Epoch: 0004 train_loss =  1.77976 val_roc_score =  0.87737 average_precision_score =  0.86656 time= 0.79843
    Epoch: 0005 train_loss =  1.76685 val_roc_score =  0.87759 average_precision_score =  0.86664 time= 0.79843
    Epoch: 0006 train_loss =  1.71661 val_roc_score =  0.87767 average_precision_score =  0.86667 time= 0.80479
    Epoch: 0007 train_loss =  1.67656 val_roc_score =  0.87775 average_precision_score =  0.86670 time= 0.80509
    Epoch: 0008 train_loss =  1.62324 val_roc_score =  0.87785 average_precision_score =  0.86679 time= 0.80446
    Epoch: 0009 train_loss =  1.57730 val_roc_score =  0.87781 average_precision_score =  0.86680 time= 0.80424
    Epoch: 0010 train_loss =  1.51882 val_roc_score =  0.87789 average_precision_score =  0.86675 time= 0.80852
    Epoch: 0011 train_loss =  1.46346 val_roc_score =  0.87792 average_precision_score =  0.86678 time= 0.80625
    Epoch: 0012 train_loss =  1.37688 val_roc_score =  0.87795 average_precision_score =  0.86684 time= 0.80474
    Epoch: 0013 train_loss =  1.31243 val_roc_score =  0.87795 average_precision_score =  0.86685 time= 0.80574
    Epoch: 0014 train_loss =  1.25133 val_roc_score =  0.87791 average_precision_score =  0.86677 time= 0.80267
    Epoch: 0015 train_loss =  1.19762 val_roc_score =  0.87802 average_precision_score =  0.86693 time= 0.80540
    Epoch: 0016 train_loss =  1.15079 val_roc_score =  0.87812 average_precision_score =  0.86698 time= 0.80784
    Epoch: 0017 train_loss =  1.09600 val_roc_score =  0.87802 average_precision_score =  0.86688 time= 0.79920
    Epoch: 0018 train_loss =  1.05011 val_roc_score =  0.87820 average_precision_score =  0.86711 time= 0.80777
    Epoch: 0019 train_loss =  1.00610 val_roc_score =  0.87840 average_precision_score =  0.86714 time= 0.80412
    Epoch: 0020 train_loss =  0.95014 val_roc_score =  0.87838 average_precision_score =  0.86713 time= 0.80210
    
    test roc score: 0.8814614254330005
    test ap score: 0.8708329314774368
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
    Epoch: 0001 train_loss =  2.08252 val_roc_score =  0.75422 average_precision_score =  0.66179 time= 0.80230
    Epoch: 0002 train_loss =  2.03940 val_roc_score =  0.86953 average_precision_score =  0.85636 time= 0.79571
    Epoch: 0003 train_loss =  2.00348 val_roc_score =  0.87872 average_precision_score =  0.86847 time= 0.79245
    Epoch: 0004 train_loss =  1.97120 val_roc_score =  0.87997 average_precision_score =  0.86995 time= 0.79640
    Epoch: 0005 train_loss =  1.93477 val_roc_score =  0.88017 average_precision_score =  0.87027 time= 0.79548
    Epoch: 0006 train_loss =  1.89215 val_roc_score =  0.88046 average_precision_score =  0.87038 time= 0.79972
    Epoch: 0007 train_loss =  1.84537 val_roc_score =  0.88072 average_precision_score =  0.87058 time= 0.79561
    Epoch: 0008 train_loss =  1.78754 val_roc_score =  0.88063 average_precision_score =  0.87049 time= 0.79802
    Epoch: 0009 train_loss =  1.72469 val_roc_score =  0.88053 average_precision_score =  0.87043 time= 0.79486
    Epoch: 0010 train_loss =  1.65402 val_roc_score =  0.88063 average_precision_score =  0.87049 time= 0.79423
    Epoch: 0011 train_loss =  1.57884 val_roc_score =  0.88052 average_precision_score =  0.87045 time= 0.79348
    Epoch: 0012 train_loss =  1.49870 val_roc_score =  0.88049 average_precision_score =  0.87046 time= 0.79649
    Epoch: 0013 train_loss =  1.42083 val_roc_score =  0.88056 average_precision_score =  0.87046 time= 0.79063
    Epoch: 0014 train_loss =  1.34764 val_roc_score =  0.88060 average_precision_score =  0.87056 time= 0.79889
    Epoch: 0015 train_loss =  1.27635 val_roc_score =  0.88038 average_precision_score =  0.87043 time= 0.79485
    Epoch: 0016 train_loss =  1.20521 val_roc_score =  0.88050 average_precision_score =  0.87058 time= 0.79927
    Epoch: 0017 train_loss =  1.13763 val_roc_score =  0.88035 average_precision_score =  0.87045 time= 0.79072
    Epoch: 0018 train_loss =  1.07326 val_roc_score =  0.88035 average_precision_score =  0.87049 time= 0.79284
    Epoch: 0019 train_loss =  1.01548 val_roc_score =  0.88023 average_precision_score =  0.87044 time= 0.78869
    Epoch: 0020 train_loss =  0.96069 val_roc_score =  0.88014 average_precision_score =  0.87037 time= 0.79441
   
    test roc score: 0.8798092171308727
    test ap score: 0.8700487009596252
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
    Epoch: 0001 train_loss =  1.83611 val_roc_score =  0.73571 average_precision_score =  0.62940 time= 0.81406
    Epoch: 0002 train_loss =  1.83237 val_roc_score =  0.87094 average_precision_score =  0.85831 time= 0.81499
    Epoch: 0003 train_loss =  1.82761 val_roc_score =  0.87429 average_precision_score =  0.86431 time= 0.81297
    Epoch: 0004 train_loss =  1.78672 val_roc_score =  0.87509 average_precision_score =  0.86525 time= 0.80870
    Epoch: 0005 train_loss =  1.76815 val_roc_score =  0.87523 average_precision_score =  0.86550 time= 0.81497
    Epoch: 0006 train_loss =  1.72495 val_roc_score =  0.87523 average_precision_score =  0.86551 time= 0.81070
    Epoch: 0007 train_loss =  1.69047 val_roc_score =  0.87593 average_precision_score =  0.86601 time= 0.80948
    Epoch: 0008 train_loss =  1.63153 val_roc_score =  0.87573 average_precision_score =  0.86593 time= 0.80709
    Epoch: 0009 train_loss =  1.57143 val_roc_score =  0.87551 average_precision_score =  0.86580 time= 0.80653
    Epoch: 0010 train_loss =  1.50240 val_roc_score =  0.87587 average_precision_score =  0.86594 time= 0.81233
    Epoch: 0011 train_loss =  1.44139 val_roc_score =  0.87567 average_precision_score =  0.86589 time= 0.80861
    Epoch: 0012 train_loss =  1.37266 val_roc_score =  0.87557 average_precision_score =  0.86571 time= 0.80932
    Epoch: 0013 train_loss =  1.32811 val_roc_score =  0.87578 average_precision_score =  0.86597 time= 0.80686
    Epoch: 0014 train_loss =  1.30064 val_roc_score =  0.87607 average_precision_score =  0.86603 time= 0.80962
    Epoch: 0015 train_loss =  1.25788 val_roc_score =  0.87592 average_precision_score =  0.86611 time= 0.80796
    Epoch: 0016 train_loss =  1.23810 val_roc_score =  0.87607 average_precision_score =  0.86617 time= 0.80750
    Epoch: 0017 train_loss =  1.18570 val_roc_score =  0.87594 average_precision_score =  0.86613 time= 0.80911
    Epoch: 0018 train_loss =  1.14961 val_roc_score =  0.87607 average_precision_score =  0.86626 time= 0.81035
    Epoch: 0019 train_loss =  1.10372 val_roc_score =  0.87593 average_precision_score =  0.86598 time= 0.81094
    Epoch: 0020 train_loss =  1.05262 val_roc_score =  0.87605 average_precision_score =  0.86613 time= 0.81442
   
    test roc score: 0.8758194438300309
    test ap score: 0.8629482273490456
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
    Epoch: 0001 train_loss =  3.24637 val_roc_score =  0.77403 average_precision_score =  0.68203 time= 0.81267
    Epoch: 0002 train_loss =  3.21157 val_roc_score =  0.87269 average_precision_score =  0.86088 time= 0.81181
    Epoch: 0003 train_loss =  3.15047 val_roc_score =  0.87391 average_precision_score =  0.86203 time= 0.81182
    Epoch: 0004 train_loss =  3.08302 val_roc_score =  0.87457 average_precision_score =  0.86271 time= 0.81055
    Epoch: 0005 train_loss =  3.03024 val_roc_score =  0.87410 average_precision_score =  0.86226 time= 0.81125
    Epoch: 0006 train_loss =  2.95011 val_roc_score =  0.87450 average_precision_score =  0.86264 time= 0.81162
    Epoch: 0007 train_loss =  2.82191 val_roc_score =  0.87460 average_precision_score =  0.86275 time= 0.81088
    Epoch: 0008 train_loss =  2.73079 val_roc_score =  0.87442 average_precision_score =  0.86256 time= 0.80648
    Epoch: 0009 train_loss =  2.61711 val_roc_score =  0.87454 average_precision_score =  0.86268 time= 0.81021
    Epoch: 0010 train_loss =  2.50720 val_roc_score =  0.87480 average_precision_score =  0.86288 time= 0.80921
    Epoch: 0011 train_loss =  2.42761 val_roc_score =  0.87506 average_precision_score =  0.86298 time= 0.81137
    Epoch: 0012 train_loss =  2.36874 val_roc_score =  0.87497 average_precision_score =  0.86282 time= 0.81466
    Epoch: 0013 train_loss =  2.29911 val_roc_score =  0.87504 average_precision_score =  0.86291 time= 0.81193
    Epoch: 0014 train_loss =  2.21190 val_roc_score =  0.87526 average_precision_score =  0.86297 time= 0.80965
    Epoch: 0015 train_loss =  2.12611 val_roc_score =  0.87511 average_precision_score =  0.86290 time= 0.81013
    Epoch: 0016 train_loss =  2.03527 val_roc_score =  0.87528 average_precision_score =  0.86314 time= 0.81365
    Epoch: 0017 train_loss =  1.96965 val_roc_score =  0.87524 average_precision_score =  0.86309 time= 0.81125
    Epoch: 0018 train_loss =  1.90381 val_roc_score =  0.87515 average_precision_score =  0.86312 time= 0.80971
    Epoch: 0019 train_loss =  1.85955 val_roc_score =  0.87487 average_precision_score =  0.86288 time= 0.80996
    Epoch: 0020 train_loss =  1.81664 val_roc_score =  0.87483 average_precision_score =  0.86293 time= 0.81270

    test roc score: 0.8826745834179653
    test ap score: 0.8715261230395998
    ```
      
    (2).预测

    ```
    from src.graph_att_gan.predict import Predict
  
    predict = Predict()
    predict.load_model_adj('config_cfg')
    # 会返回原始的图邻接矩阵和经过模型编码后的hidden embedding经过内积解码的邻接矩阵，可以对这两个矩阵进行比对，得出link prediction.
    adj_orig, adj_rec = predict.predict()
    ```
  
  ##### 5.NHGATModelVAE(src/graph_nheads_att_gae)
    (1).训练
    ```
    from src.graph_nheads_att_gae.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
    ```
    Epoch: 0001 train_loss =  1.85570 val_roc_score =  0.80750 average_precision_score =  0.72917 time= 0.84645
    Epoch: 0002 train_loss =  1.78607 val_roc_score =  0.88103 average_precision_score =  0.87114 time= 0.84186
    Epoch: 0003 train_loss =  1.68021 val_roc_score =  0.88117 average_precision_score =  0.87144 time= 0.84135
    Epoch: 0004 train_loss =  1.52555 val_roc_score =  0.88115 average_precision_score =  0.87141 time= 0.84212
    Epoch: 0005 train_loss =  1.38254 val_roc_score =  0.88070 average_precision_score =  0.87098 time= 0.83917
    Epoch: 0006 train_loss =  1.40003 val_roc_score =  0.88106 average_precision_score =  0.87134 time= 0.84185
    Epoch: 0007 train_loss =  1.31239 val_roc_score =  0.88081 average_precision_score =  0.87110 time= 0.83766
    Epoch: 0008 train_loss =  1.17827 val_roc_score =  0.88102 average_precision_score =  0.87134 time= 0.84063
    Epoch: 0009 train_loss =  1.08710 val_roc_score =  0.88086 average_precision_score =  0.87126 time= 0.84173
    Epoch: 0010 train_loss =  1.01816 val_roc_score =  0.88136 average_precision_score =  0.87162 time= 0.84121
    Epoch: 0011 train_loss =  0.95128 val_roc_score =  0.88128 average_precision_score =  0.87133 time= 0.84128
    Epoch: 0012 train_loss =  0.87212 val_roc_score =  0.88127 average_precision_score =  0.87142 time= 0.84218
    Epoch: 0013 train_loss =  0.80497 val_roc_score =  0.88134 average_precision_score =  0.87154 time= 0.84077
    Epoch: 0014 train_loss =  0.75538 val_roc_score =  0.88088 average_precision_score =  0.87120 time= 0.83701
    Epoch: 0015 train_loss =  0.70903 val_roc_score =  0.88063 average_precision_score =  0.87073 time= 0.83698
    Epoch: 0016 train_loss =  0.68525 val_roc_score =  0.88035 average_precision_score =  0.87055 time= 0.83837
    Epoch: 0017 train_loss =  0.66079 val_roc_score =  0.87995 average_precision_score =  0.87053 time= 0.83806
    Epoch: 0018 train_loss =  0.65187 val_roc_score =  0.87924 average_precision_score =  0.86958 time= 0.84210
    Epoch: 0019 train_loss =  0.64572 val_roc_score =  0.87929 average_precision_score =  0.86995 time= 0.84069
    Epoch: 0020 train_loss =  0.64103 val_roc_score =  0.87951 average_precision_score =  0.87026 time= 0.83967

    test roc score: 0.877033361471422
    test ap score: 0.867286248500891
    ```
      
    (2).预测

    ```
    from src.graph_nheads_att_gae.predict import Predict
  
    predict = Predict()
    predict.load_model_adj('config_cfg')
    # 会返回原始的图邻接矩阵和经过模型编码后的hidden embedding经过内积解码的邻接矩阵，可以对这两个矩阵进行比对，得出link prediction.
    adj_orig, adj_rec = predict.predict()
    ```
  
    ##### 6.NHGATModelGAN(src/graph_nheads_att_gan)
    (1).训练
    ```
    from src.graph_nheads_att_gan.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
    ```
    Epoch: 0001 train_loss =  3.24091 val_roc_score =  0.77050 average_precision_score =  0.66992 time= 0.85475
    Epoch: 0002 train_loss =  3.18022 val_roc_score =  0.87671 average_precision_score =  0.86657 time= 0.84643
    Epoch: 0003 train_loss =  3.09047 val_roc_score =  0.87715 average_precision_score =  0.86704 time= 0.84354
    Epoch: 0004 train_loss =  2.95696 val_roc_score =  0.87695 average_precision_score =  0.86698 time= 0.84279
    Epoch: 0005 train_loss =  2.87052 val_roc_score =  0.87747 average_precision_score =  0.86741 time= 0.84714
    Epoch: 0006 train_loss =  2.88739 val_roc_score =  0.87742 average_precision_score =  0.86727 time= 0.84777
    Epoch: 0007 train_loss =  2.78251 val_roc_score =  0.87757 average_precision_score =  0.86748 time= 0.84134
    Epoch: 0008 train_loss =  2.65458 val_roc_score =  0.87766 average_precision_score =  0.86745 time= 0.84429
    Epoch: 0009 train_loss =  2.60484 val_roc_score =  0.87798 average_precision_score =  0.86780 time= 0.84680
    Epoch: 0010 train_loss =  2.56642 val_roc_score =  0.87806 average_precision_score =  0.86766 time= 0.84952
    Epoch: 0011 train_loss =  2.49832 val_roc_score =  0.87826 average_precision_score =  0.86771 time= 0.84535
    Epoch: 0012 train_loss =  2.38511 val_roc_score =  0.87799 average_precision_score =  0.86763 time= 0.84903
    Epoch: 0013 train_loss =  2.28920 val_roc_score =  0.87781 average_precision_score =  0.86762 time= 0.84161
    Epoch: 0014 train_loss =  2.23039 val_roc_score =  0.87791 average_precision_score =  0.86761 time= 0.84422
    Epoch: 0015 train_loss =  2.14044 val_roc_score =  0.87782 average_precision_score =  0.86750 time= 0.84063
    Epoch: 0016 train_loss =  2.05134 val_roc_score =  0.87774 average_precision_score =  0.86754 time= 0.84043
    Epoch: 0017 train_loss =  1.95402 val_roc_score =  0.87745 average_precision_score =  0.86740 time= 0.84461
    Epoch: 0018 train_loss =  1.89405 val_roc_score =  0.87714 average_precision_score =  0.86720 time= 0.84435
    Epoch: 0019 train_loss =  1.83182 val_roc_score =  0.87690 average_precision_score =  0.86693 time= 0.84567
    Epoch: 0020 train_loss =  1.74144 val_roc_score =  0.87683 average_precision_score =  0.86717 time= 0.84130

    test roc score: 0.8767371798715641
    test ap score: 0.8680650766563964
    ```
      
    (2).预测

    ```
    from src.graph_nheads_att_gan.predict import Predict
  
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
