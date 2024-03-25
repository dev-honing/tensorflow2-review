# main.py

#* 1. Tensorflow2 설치 및 버전 정보 확인
import tensorflow as tf
# print("TensorFlow version:", tf.__version__) # TensorFlow version: 2.16.1

#* 2. Tensorflow2 MNIST 데이터셋 다운로드
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist # keras 데이터셋 중에서 MNIST 데이터셋을 변수에 할당

#* 3. MNIST 데이터셋을 훈련용 데이터와 테스트 데이터로 나누기
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 데이터셋 로드
x_train, x_test = x_train / 255.0, x_test / 255.0 

#? 데이터를 나누는 이유?
# 훈련 데이터로 모델을 학습시킨 후, 테스트 데이터로 모델을 평가하기 위함

# 4. 각 데이터셋에 차원 추가 및 데이터 타입 변경
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
