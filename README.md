# gg1th_dl_ex

TensorFlow/Keras 기반 딥러닝 실습 정리. DNN(이진·다중 분류, 회귀)부터 CNN, 전이학습(ResNet50)까지 순서대로 다룬다.

## 환경

- Python 3.12, TensorFlow (GPU / WSL2)
- 패키지 관리: [uv](https://github.com/astral-sh/uv) (`pyproject.toml`, `uv.lock`)
- GPU 셋업은 [`GPU_SETUP.md`](GPU_SETUP.md) 참고

```bash
uv sync
uv run python check_gpu.py   # GPU 인식 확인
```

## 노트북

### DNN
| 파일 | 내용 |
|------|------|
| `1.DNN_이진분류_피마_인디언.ipynb` | 이진 분류 (Pima Indians Diabetes) |
| `2.DNN_이진분류_Churn_Modelling.ipynb` | 이진 분류 (고객 이탈 예측) |
| `3.DNN_다중_분류_문제_iris.ipynb` | 다중 분류 (Iris) |
| `4.DNN_회귀_Car_Purchase_Amount_Predictions_솔루션.ipynb` | 회귀 (구매액 예측) |
| `5.DNN_회귀_자동차연비예측_ANN_솔루션.ipynb` | 회귀 (자동차 연비 예측) |
| `6_DNN_Building_fashion_mnist.ipynb` | Fashion-MNIST (DNN) |

### CNN
| 파일 | 내용 |
|------|------|
| `7_CNN_fashion_mnist.ipynb` | Fashion-MNIST (CNN) |
| `8.CNN_cat&dog_Augmentation.ipynb` | 개·고양이 분류 + 데이터 증강 |

### 전이학습 (Transfer Learning)
| 파일 | 내용 |
|------|------|
| `9.Transfer_Learning_ResNet50.ipynb` | ResNet50 전이학습 기초 |
| `10-1.CNN_Transfer_Learning_softmax.ipynb` | 전이학습 + softmax |
| `10-2.CNN_Fine_Tunning_softmax.ipynb` | 파인튜닝 |
| `11.CNN_ResNet50_전이학습2_문제해결.ipynb` | ResNet50 전이학습 문제해결 |

## 데이터 / 모델

`datas_dnn/`의 작은 CSV만 저장소에 포함되며, 대용량 이미지 데이터셋(`image_data/`, `cats_and_dogs_filtered` 등)과 학습된 모델(`models/`)은 용량 문제로 제외했다(`.gitignore` 참고).
