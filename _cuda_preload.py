"""pip 로 깔린 NVIDIA CUDA 라이브러리를 인터프리터 시작 시점에 미리 로드한다.

[왜 필요한가]
TensorFlow(tensorflow[and-cuda])는 CUDA 라이브러리를 "libcudnn.so.9" 같은
bare soname(순수 파일명)으로 dlopen 한다. 그런데 nvidia-*-cu12 wheel 들은 그
파일들을 site-packages/nvidia/<패키지>/lib 밑에 설치하는데, 이 경로는 리눅스
동적 로더의 검색 경로에 들어있지 않다. 그래서 아무 조치 없이 TF 를 import 하면
"Cannot dlopen some GPU libraries ..." 경고와 함께 GPU 가 스킵된다.

[이 파일이 하는 일]
여기서 nvidia/*/lib/*.so* 를 전부 ctypes.CDLL(..., RTLD_GLOBAL) 로 미리 로드해
둔다. RTLD_GLOBAL 로 한 번 올려두면 그 라이브러리가 soname 으로 프로세스에
등록되므로, 나중에 TF 가 같은 soname 으로 dlopen 할 때 이미 로드된 객체를 바로
찾는다. 결과적으로 LD_LIBRARY_PATH 나 --env-file 없이도, 어떤 실행 방식(그냥
`uv run python`, VS Code, Jupyter 커널)에서든 GPU 가 잡힌다.

[어떻게 트리거되나]
이 파일은 venv 의 site-packages 로 복사되고, 같은 폴더의 zzz_cuda_preload.pth
가 인터프리터 시작 시마다 `import _cuda_preload` 를 실행해 이 코드를 돌린다.
(sitecustomize.py 는 시스템 /usr/lib/python3.12/sitecustomize.py 에 가려져서
동작하지 않기 때문에 .pth 방식을 쓴다.)

[재설치]
.venv 를 다시 빌드하면 이 복사본과 .pth 가 사라진다. 그때는 프로젝트 루트에서
    uv run python setup_gpu_env.py
를 실행하면 이 파일이 다시 복사되고 .pth 도 재생성된다.

nvidia 패키지가 없으면 아무 것도 하지 않는 안전한 no-op 이다.
"""
import ctypes
import glob
import os

try:
    import nvidia
except ImportError:
    nvidia = None

if nvidia is not None:
    _base = os.path.dirname(nvidia.__file__)
    _sos = sorted(glob.glob(os.path.join(_base, "*", "lib", "*.so*")))
    _pending = _sos
    for _ in range(2):  # 한 번 재시도 -> 라이브러리 간 로드 순서 의존성 제거
        _failed = []
        for _so in _pending:
            try:
                ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                _failed.append(_so)
        if not _failed:
            break
        _pending = _failed
