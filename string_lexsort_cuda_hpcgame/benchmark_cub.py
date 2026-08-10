from pathlib import Path

from judge import runtime


runtime.CUDA_SOURCE = Path(__file__).with_name("baselines") / "cub_submission.cu"


if __name__ == "__main__":
    raise SystemExit(runtime.main())
