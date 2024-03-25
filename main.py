# main.py

#* 1. Tensorflow2 설치 및 버전 정보 확인
import tensorflow as tf
# print("TensorFlow version:", tf.__version__) # TensorFlow version: 2.16.1

#* 2. Tensorflow2 MNIST 데이터셋 다운로드
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist # keras 데이터셋 중에서 MNIST 데이터셋을 변수에 할당

#* 3. MNIST 데이터셋을 훈련용 데이터와 테스트용 데이터로 나누기
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 데이터셋 로드
x_train, x_test = x_train / 255.0, x_test / 255.0 

#? 데이터를 나누는 이유?
# 훈련용 데이터로 모델을 학습시킨 후, 테스트용 데이터로 모델을 평가하기 위함

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
test_loss = tf.keras.metrics.Mean(name='test_loss') # 테스트용 손실
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy') # 테스트용 정확도

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

#* 10. 모델 훈련
EPOCHS = 5

# 지표 초기화 및 에포크 반복
for epoch in range(EPOCHS):
    # 에포크 시작 시 지표를 초기화
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    # 훈련용 데이터 세트를 순회하며 모델을 훈련
    for images, labels in train_ds:
        train_step(images, labels)
    
    # 테스트용 데이터 세트를 순회하며 모델을 평가
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    # 에포트마다 손실 및 정확도를 출력
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )

#* 11. 콘솔에 출력된 결과
"""
Administrator@User -2023BIRMN MINGW64 ~/Desktop/ho/tensorflow2-review (main)
$ python main.py
2024-03-25 17:53:41.443707: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-25 17:53:42.194401: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-25 17:53:44.020678: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-25 17:54:44.217908: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2024-03-25 17:54:44.981128: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 1, Loss: 0.12936609983444214, Accuracy: 96.15833282470703, Test Loss: 0.0566282719373703, Test Accuracy: 98.23999786376953
2024-03-25 17:55:46.018726: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2024-03-25 17:55:46.713302: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 2, Loss: 0.03984072431921959, Accuracy: 98.77166748046875, Test Loss: 0.05685282126069069, Test Accuracy: 98.05999755859375
2024-03-25 17:56:45.738726: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2024-03-25 17:56:46.741205: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 3, Loss: 0.019433794543147087, Accuracy: 99.37833404541016, Test Loss: 0.05554548278450966, Test Accuracy: 98.32999420166016
2024-03-25 17:57:46.358867: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2024-03-25 17:57:47.074266: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 4, Loss: 0.011850987561047077, Accuracy: 99.57666015625, Test Loss: 0.05386853590607643, Test Accuracy: 98.33999633789062
2024-03-25 17:58:45.727581: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2024-03-25 17:58:46.383442: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 5, Loss: 0.008456694893538952, Accuracy: 99.72833251953125, Test Loss: 0.055616337805986404, Test Accuracy: 98.63999938964844
"""