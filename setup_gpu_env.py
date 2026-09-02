"""GPU(CUDA) 환경을 이 프로젝트에 자동 설정하는 스크립트.

[언제 실행하나]
.venv 를 새로 만들거나 재빌드한 직후 한 번 실행한다:
    uv run python setup_gpu_env.py
그 뒤엔 그냥 `uv run python ...`, VS Code, Jupyter 어디서든 GPU 가 자동으로 잡힌다.

[무엇을 하나] 아래 3가지를 자동으로 해준다.
  1) .env 파일 생성 - site-packages/nvidia/*/lib 경로들을 찾아
     LD_LIBRARY_PATH=... 형태로 써준다. (.vscode/settings.json 의 python.envFile
     / UV_ENV_FILE 이 이걸 읽어 CUDA 경로를 자동 주입하는 fallback 용)
  2) _cuda_preload.py 를 venv 의 site-packages 로 복사. (실제로 GPU 라이브러리를
     미리 로드해주는 본체 -> 자세한 설명은 그 파일 상단 주석 참고)
  3) zzz_cuda_preload.pth 를 site-packages 에 생성. 내용은 `import _cuda_preload`
     한 줄. 인터프리터가 시작될 때마다 (1)의 preload 를 자동 실행시키는 트리거.
     파일명이 zzz~ 인 이유: .pth 는 이름 알파벳 순으로 실행되는데, nvidia 패키지가
     먼저 import 가능해야 하므로 가장 뒤로 밀어두기 위함.

두 겹 안전장치(.pth = 주 경로, .env = 보조)를 모두 깔아주는 셈이다.

실행 후 검증:  uv run python check_gpu.py
"""
import glob
import os
import shutil
import site
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_site_packages() -> str:
    """현재 실행 중인 (venv) 인터프리터의 site-packages 경로를 반환."""
    candidates = [p for p in site.getsitepackages() if p.endswith("site-packages")]
    return candidates[0] if candidates else site.getsitepackages()[0]


def main() -> int:
    site_packages = find_site_packages()
    nvidia_base = os.path.join(site_packages, "nvidia")

    # nvidia-*-cu12 wheel 들이 깐 lib 디렉터리들을 수집 (예: .../nvidia/cudnn/lib)
    lib_dirs = sorted(glob.glob(os.path.join(nvidia_base, "*", "lib")))
    if not lib_dirs:
        print("[!] nvidia/*/lib 경로를 못 찾았어. "
              "`uv sync` 또는 tensorflow[and-cuda] 설치부터 확인해줘.", file=sys.stderr)
        print(f"    (확인한 위치: {nvidia_base})", file=sys.stderr)
        return 1

    # 1) .env 재생성 -----------------------------------------------------------
    env_path = os.path.join(PROJECT_DIR, ".env")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("LD_LIBRARY_PATH=" + ":".join(lib_dirs) + "\n")
    print(f"[✓] .env 작성 완료 ({len(lib_dirs)}개 CUDA lib 경로)")

    # 2) _cuda_preload.py 를 venv 로 복사 --------------------------------------
    src = os.path.join(PROJECT_DIR, "_cuda_preload.py")
    dst = os.path.join(site_packages, "_cuda_preload.py")
    if not os.path.exists(src):
        print(f"[!] {src} 가 없어. 프로젝트에 _cuda_preload.py 부터 두어야 해.",
              file=sys.stderr)
        return 1
    shutil.copyfile(src, dst)
    print(f"[✓] _cuda_preload.py -> {dst}")

    # 3) .pth 트리거 생성 ------------------------------------------------------
    pth_path = os.path.join(site_packages, "zzz_cuda_preload.pth")
    with open(pth_path, "w", encoding="utf-8") as f:
        f.write("import _cuda_preload\n")
    print(f"[✓] {pth_path}")

    print("\n완료! 검증:  uv run python check_gpu.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
