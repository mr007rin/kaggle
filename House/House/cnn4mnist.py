'''
This script performs the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from mnist

This script uses cnn to predict image's values
'''

# Remember to update the script for the new data when you change this URL

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# 设置随机数种子
seed = 7
numpy.random.seed(seed)

# =====================================================================
def baseline_model():
    # 创建模型
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # 加载(下载，如果需要的话)MNIST数据集
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #  转换数据为 [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2]).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1,X_test.shape[1],X_test.shape[2]).astype('float32')

    # 规范化输入从 0-255 到 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # 独热编码
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # 构造
    model = baseline_model()
    # 调整
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    # 最终输出
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))