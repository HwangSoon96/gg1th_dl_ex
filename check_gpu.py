"""GPU 세팅이 제대로 됐는지 검증하는 스크립트.

사용:  uv run python check_gpu.py

[무엇을 확인하나]
  1) TensorFlow 가 물리 GPU 를 인식하는지 (list_physical_devices('GPU'))
  2) GPU 에서 실제 연산이 도는지 (작은 행렬곱)
  3) cuDNN 경로까지 정상인지 (작은 Conv2D 순전파 1회) - "Cannot dlopen
     some GPU libraries" 류 문제가 남아있으면 여기서 걸린다.

셋 다 통과하면 preload(.pth) / .env 세팅이 정상이라는 뜻.
GPU 가 안 잡히면 먼저  uv run python setup_gpu_env.py  를 돌려봐.
"""
import sys

import tensorflow as tf

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

if not gpus:
    print("\n[✗] GPU 미인식. `uv run python setup_gpu_env.py` 실행 후 다시 시도해줘.")
    sys.exit(1)

# 1) GPU 위에서 실제 연산 -----------------------------------------------------
with tf.device("/GPU:0"):
    a = tf.random.normal((512, 512))
    b = tf.random.normal((512, 512))
    c = tf.matmul(a, b)
print("[✓] GPU 행렬곱 OK, result shape =", c.shape)

# 2) cuDNN 경로 검증 (Conv2D 한 번) -------------------------------------------
with tf.device("/GPU:0"):
    x = tf.random.normal((1, 28, 28, 1))
    conv = tf.keras.layers.Conv2D(8, 3, activation="relu")
    y = conv(x)
print("[✓] cuDNN Conv2D OK, output shape =", y.shape)

print("\n모든 검증 통과 — GPU + cuDNN 정상 동작 ✅")
