# 快速自检（CPU/GPU 可用性）
import torch,sys;print(sys.version);
print(torch.__version__);
print('CUDA?', torch.cuda.is_available())
