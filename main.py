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

#* 4. 각 데이터셋에 차원 추가 및 데이터 타입 변경
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

#* 5. 데이터셋 섞기 및 배치 설정
# 섞기
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

#? 섞는 이유?
# 특정한 순서로 정렬되어 있다면 그 순서에 편향성을 가질 우려가 있으므로 독립적으로 학습시키기 위함

# 배치 설정
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#* 6. 모델링(서브 클래스 방식) - Keras API로 커스텀 모델 정의
class MyModel(Model):
  # 모델의 레이어를 초기화
  def __init__(self):
    # 부모 클래스의 생성자를 호출
    super(MyModel, self).__init__()
    # 레이어 정의(Conv2D, Flatten, Dense)
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  # 모델의 실행 메서드 정의(순전파)
  def call(self, x):
    # 입력 데이터를 레이어에 순차적으로 전달
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# 모델 객체 생성
model = MyModel()

#* 7. 손실함수 및 최적화 알고리즘 설정
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # 손실함수

optimizer = tf.keras.optimizers.Adam() # 최적화 알고리즘

#* 8. 손실 및 정확도 지표 설정
# 훈련용 데이터에 대한 평균 손실 및 정확도 지표 설정
train_loss = tf.keras.metrics.Mean(name='train_loss') # 훈련 손실
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy') # 훈련 정확도

# 테스트용 데이터에 대한 평균 손실 및 정확도 지표 설정
test_loss = tf.keras.metrics.Mean(name='test_loss') # 테스트 손실
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy') # 테스트 정확도

#* 9. 스텝 함수 정의 - 훈련용 및 테스트용
@tf.function
# 훈련용 스텝 함수: 주어진 이미지 및 라벨로 모델을 훈련하고 손실 및 성능 지표 업데이트
def train_step(images, labels):
    # 계산을 위한 Gradient 컨텍스트 생성
    with tf.GradientTape() as tape:
        # 모델에 이미지 전달 및 예측
        predictions = model(images, training=True)
        # 손실 계산
        loss = loss_object(labels, predictions)
    
    # 손실에 대한 모델의 그래디언트 계산
    gradients = tape.gradient(loss, model.trainable_variables)

    # 옵티마이저를 사용해 가중치 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 훈련용 데이터에 대한 손실 및 정확도 지표 업데이트
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
# 테스트용 스텝 함수: 주어진 이미지 및 라벨로 모델을 평가하고 손실 및 성능 지표 업데이트
def test_step(images, labels):
    # 모델에 이미지 전달 및 예측
    predictions = model(images, training=False)
    # 손실 계산
    t_loss = loss_object(labels, predictions)

    # 테스트용 데이터에 대한 손실 및 정확도 지표 업데이트
    test_loss(t_loss)
    test_accuracy(labels, predictions)
