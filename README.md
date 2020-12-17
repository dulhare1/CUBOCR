# CUBOCR

TensorFlow-deep learning框架
Keras-類似介面，可在TensorFlow上執行(須先install)，方便使用可快速實現deep learning，但細節參數微調能力較差

https://keras.io/api/applications/
使用圖片分類模型ResNet50，硬體需求較低，可在ＣＰＵ上執行

##------------------------------------------------------------------------------------------------
data:訓練的資料圖
predicting.py：讀取欲辨別的圖片，call訓練好的model（weights_data夾內）
training_tf.py:training function


所有樣本共150000張圖片，epoch設15->一次取10000張，batch size設100->10000(張)/100(size)=100(次)->一次epoch跑100次

