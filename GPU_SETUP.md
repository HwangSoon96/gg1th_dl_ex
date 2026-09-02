# GPU(CUDA) 환경 설정 안내

이 프로젝트에서 TensorFlow 가 **아무 플래그·환경변수 없이** GPU 를 잡도록 만들어 둔
세팅에 대한 설명이다. (WSL2 Ubuntu 24.04 / RTX 4060 Laptop, CC 8.9)

## 왜 이 세팅이 필요한가

`tensorflow[and-cuda]` 는 CUDA 라이브러리를 `libcudnn.so.9` 같은 **bare soname**
(순수 파일명)으로 `dlopen` 한다. 그런데 `nvidia-*-cu12` wheel 들은 그 `.so` 파일을
`.venv/lib/python3.12/site-packages/nvidia/<패키지>/lib/` 밑에 설치하는데, 이 경로는
리눅스 동적 로더의 기본 검색 경로에 **없다.**

그래서 그냥 `import tensorflow` 하면:

```
Cannot dlopen some GPU libraries ... Skipping registering GPU devices...
```

경고와 함께 **GPU 가 CPU 로 폴백**된다.

## 해결 구조 (두 겹)

### ① 주 경로 — `.pth` 자동 preload  ✅ 권장

- **`_cuda_preload.py`** : `nvidia/*/lib/*.so*` 를 전부 `ctypes.CDLL(..., RTLD_GLOBAL)`
  로 **미리 로드**한다. 한 번 RTLD_GLOBAL 로 올려두면 soname 으로 프로세스에 등록되어,
  이후 TF 가 같은 soname 으로 `dlopen` 할 때 이미 로드된 객체를 바로 찾는다.
- **`zzz_cuda_preload.pth`** (venv site-packages 안, 내용은 `import _cuda_preload` 한 줄):
  인터프리터가 시작될 때마다 위 preload 를 자동 실행시키는 트리거.
  - `.pth` 는 이름 알파벳 순으로 실행되므로 `nvidia` 패키지가 먼저 import 가능하도록
    `zzz~` 로 이름 지어 맨 뒤로 밀어둔다.
  - `sitecustomize.py` 방식은 시스템의 `/usr/lib/python3.12/sitecustomize.py` 에
    가려져서 **동작하지 않는다.** 그래서 `.pth` 방식을 쓴다.

→ 이 덕분에 **그냥 `uv run python ...`, VS Code, Jupyter 커널** 어디서든 플래그 없이 GPU 가 잡힌다.

### ② 보조 경로 — `.env` fallback

- **`.env`** : `LD_LIBRARY_PATH=<nvidia */lib 경로들>` 를 담고 있다.
- **`.vscode/settings.json`** 이 `python.envFile` 과
  `terminal.integrated.env.linux.UV_ENV_FILE` 로 이 `.env` 를 읽어 경로를 주입한다.
- VS Code 밖에서 `.pth` 없이 돌릴 때는:  `uv run --env-file .env python ...`

## 파일 목록

| 파일 | 역할 |
|------|------|
| `_cuda_preload.py`      | GPU 라이브러리를 실제로 미리 로드하는 본체 (venv 로 복사됨) |
| `setup_gpu_env.py`      | `.env` 재생성 + `_cuda_preload.py` 복사 + `.pth` 생성 (설정 자동화) |
| `check_gpu.py`          | GPU / cuDNN 정상 동작 검증 |
| `.env`                  | `LD_LIBRARY_PATH` (보조 경로용, 자동 생성됨) |
| `.vscode/settings.json` | VS Code / 터미널에 `.env` 주입 |

> `.pth` 와 venv 안 `_cuda_preload.py` 복사본은 **`.venv` 폴더 안**에 있어 재빌드 시 사라진다.
> 위 표의 파일들은 프로젝트에 남는 **원본/재생성 도구**다.

## `.venv` 를 다시 빌드했다면 (중요)

venv 를 재생성하면 `.pth` preload 와 `.env` 가 날아간다. 프로젝트 루트에서 한 번:

```bash
uv run python setup_gpu_env.py     # .env 재작성 + .pth preload 재설치
uv run python check_gpu.py         # 검증
```

## 빠른 검증

```bash
uv run python check_gpu.py
```

기대 출력(요약):

```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
[✓] GPU 행렬곱 OK ...
[✓] cuDNN Conv2D OK ...
모든 검증 통과 — GPU + cuDNN 정상 동작 ✅
```
