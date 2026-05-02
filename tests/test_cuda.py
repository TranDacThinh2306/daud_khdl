def test_cuda():
    import torch
    import sys

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
        # Quan trọng: kiểm tra PyTorch được build với CUDA nào
    print(f"PyTorch CUDA version used at build time: {torch.version.cuda}")

test_cuda()