import math
import numpy as np
import os
import pandas as pd
import time
import torch
from tqdm.auto import tqdm

from datasets import WikiText2, DatasetSplit
from model.language_model import MambaLM
from configs.config import MambaConfig
from utils.train_utils import *
from utils.test_utils import *
from utils.checkpoints import *

def train_model(model, train_data, valid_data, args):
    parts = os.path.splitext(args.savefilename)
    savefilename_last = f"{parts[0]}__last{parts[1]}"
    savefilename_best = f"{parts[0]}__best{parts[1]}"
    logs_train = []
    logs_test = []
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    print('Initiating training, {} iterations/epoch.'.format(len(train_loader)))
    log_interval = max(10, len(train_loader) // 20)
    optimizer = setup_optimizer_from_args(args, model)
    scheduler = build_scheduler(args, optimizer)
    curr_lr = optimizer.param_groups[0]['lr']

    # Logging utils
    best_val_loss = float('inf')
    def _log_train():
        nonlocal logs_train
        
        cur_loss = total_loss / log_interval
        elapsed = time.time() - t0
        tqdm.write(f"{epoch:3d} "f"({100*i/len(train_loader):5.1f}%) "f"{elapsed*1000/log_interval:10.2f} " f"{curr_lr:10.4e} "
               f"{cur_loss:10.4f} "f"{math.exp(cur_loss):12.2f} "f"{format_time(elapsed):>10}")
        logs_train.append(dict(epoch=epoch+i*1.0/len(train_loader), lr=curr_lr, loss=cur_loss, ppl=math.exp(cur_loss)))
        save_df(pd.DataFrame(logs_train),f"{args.logfilename}_train.csv",)

    def _log_valid():
        nonlocal logs_test, best_val_loss
        t0 = time.time()
        val_metrics = evaluate_model(model, valid_data, args.batch_size)
        print('-' * 100)
        print('| checkpoint | epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} | '
                'validation perplexity {:8.2f}'.format(epoch, (time.time() - t0), val_metrics['loss'], val_metrics['ppl']))
        val_metrics['epoch'] = epoch
        logs_test.append(val_metrics)
        save_df(pd.DataFrame(logs_test),f"{args.logfilename}_val.csv",)
        print('-' * 100)
        print(f"{'epoch':>3} "f"{'%':>7} "f"{'ms/batch':>10} "f"{'lr':>10} "f"{'loss':>10} "f"{'ppl':>12} "f"{'elapsed':>10}")

        # Save latest checkpoint
        save_checkpoint(model,optimizer,epoch,best_val_loss,savefilename_last,)
        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model,optimizer,epoch,best_val_loss,savefilename_best,)

    # Training Loop
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(enumerate(train_loader),total=len(train_loader),desc=f"Epoch {epoch+1}/{args.num_epochs}",leave=True,)
        for i, (x, y) in pbar:
            if i % log_interval == 0 and i > 0:
                _log_train()
                total_loss = 0
            x = x.to(device)
            y = y.to(device)    
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)),y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1} | LR = "f"{optimizer.param_groups[0]['lr']:.2e}")
        _log_valid()
    return model

if __name__ == '__main__':
    parser = make_train_parser()
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    device = get_device_from_arg(args.device)

    dataset_args = dict(data_dir=args.data_dir, context_size=args.context_size, use_block_split=True)
    train_data = WikiText2(**dataset_args, split=DatasetSplit.train, max_vocab_size=args.max_vocab_size)
    valid_data = WikiText2(**dataset_args, split=DatasetSplit.valid, max_vocab_size=args.max_vocab_size)
    test_data = WikiText2(**dataset_args, split=DatasetSplit.test, max_vocab_size=args.max_vocab_size)

    # Setup model and updating Vocabulary size based on training data.
    config = MambaConfig(d_model=args.d_model,n_layer=args.n_layer,d_state=args.d_state,dropout=args.dropout,)
    config.vocab_size = len(train_data.word2idx)
    model = MambaLM(config,device=device,)
    print("Model device:", next(model.parameters()).device)
    count = count_parameters(model)
    print('Initialized model with {} parameters'.format(count))

    try:
        train_model(model, train_data, valid_data, args)
    except KeyboardInterrupt:
        print('Graceful Exit')

    print('Restoring best checkpointed model...')
    parts = os.path.splitext(args.savefilename)
    savefilename_best = f'{parts[0]}__best{parts[1]}'
    model = load_model_weights(model,path=savefilename_best,device=device,)

    test_metrics = evaluate_model(model, test_data, args.batch_size)
    print('=' * 89)
    print('| end of training | test loss {:5.2f} | test perplexity {:8.2f}'.format(test_metrics['loss'], test_metrics['ppl']))
    print(test_metrics)
    print('=' * 89)


    # print('\nUncurated samples')
    # print('-' * 89)
    # for i in range(5):
    #     print('({})'.format(i), utils.sample_from_model(model, train_data))
