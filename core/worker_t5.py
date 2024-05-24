# import torch
# import torch.optim
# import torch.utils.data
# import torch.nn.parallel
# from core.train_t5 import *
# from utils import *
# from core.network_t5 import get_model
# from loss import get_criterion
# from datasets import get_dataloader
# from transformers import DataCollatorForSeq2Seq

# def main_worker(args):
#     args.logger = initialize_logger(args)
#     train_loader, train_sampler, val_loader, src_lang, tgt_lang = get_dataloader(args)
#     model = get_model(args, src_lang, tgt_lang).cuda()
#     optimizer = get_optimizer(args, model)
#     scheduler = get_scheduler(args, optimizer)
#     criterion = get_criterion(args)
#     start_epoch = 0

#     if not args.resume_model == '':
#         resume_model_dict = model.load_model(args.resume_model)
#         optimizer.load_state_dict(resume_model_dict['optimizer'])
#         scheduler.load_state_dict(resume_model_dict['scheduler'])
#         start_epoch = resume_model_dict["epoch"] + 1
#         args.logger.info("The whole model has been loaded from " + args.resume_model)
#         args.logger.info("The model resumes from epoch " + str(resume_model_dict["epoch"]))
#     else:
#         args.logger.info("The model is trained from scratch")

#     model = torch.nn.parallel.DistributedDataParallel(
#         model, 
#         device_ids=[args.local_rank], 
#         output_device=args.local_rank, 
#         find_unused_parameters=True
#     )

#     data_collator = DataCollatorForSeq2Seq(tokenizer=args.tokenizer, model=model)

#     for epoch in range(start_epoch, args.max_epoch):
#         train_sampler.set_epoch(epoch)
#         loss = train(args, epoch, train_loader, model, criterion, optimizer, data_collator, scheduler)
#         args.logger.info("----------Epoch:{:>3d}, training loss is {:>5.4f} ---------".format(epoch, loss))
#         if epoch > 0 and (epoch % args.eval_epoch == 0 or epoch >= args.max_epoch - 5):
#             is_best = False
#             if args.local_rank == 0:
#                 save_checkpoint({
#                     'epoch': epoch,
#                     'state_dict': model.module.state_dict(),
#                     'scheduler': scheduler.state_dict(),
#                     'optimizer': optimizer.state_dict()
#                 }, is_best, args.dump_path)
#         scheduler.step()
    
#     args.logger.info("------------------- Train Finished -------------------")


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from core.train import get_model, get_optimizer, get_criterion, train, save_checkpoint
import torch.backends.cudnn as cudnn
import random
from datasets import get_dataloader

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    train_loader, train_sampler, val_loader, src_lang, tgt_lang = get_dataloader(args)
    model = get_model(args, src_lang, tgt_lang).cuda(args.gpu)
    optimizer = get_optimizer(args, model)
    total_steps = len(train_loader) * args.max_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    criterion = get_criterion(args)
    start_epoch = 0

    if args.resume_model:
        checkpoint = torch.load(args.resume_model, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    for epoch in range(start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)
        loss = train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        print(f"Epoch {epoch}, Loss {loss}")
        if epoch % args.eval_epoch == 0 or epoch >= args.max_epoch - 5:
            if args.rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best=False, filename=f'checkpoint_{epoch}.pth.tar')
        scheduler.step()
    print("Training Finished")
