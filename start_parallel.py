import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import *
from core.worker_t5 import main_worker
# from core.train_t5 import main_worker
import torch.multiprocessing as mp

def main():
    args = get_parser()
    args.world_size = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))

if __name__ == '__main__':
    main()