from pathlib import Path
import sys
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E/'direct_implementation'))
from shared_gpu_policy import install_priority
install_priority()
import torch
torch.cuda.set_per_process_memory_fraction(16*1024**3/torch.cuda.get_device_properties(0).total_memory)
import train
if __name__=='__main__':train.main()
