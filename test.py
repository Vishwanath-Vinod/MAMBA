import math
import numpy as np
import os
import pandas as pd
import time
import torch

from datasets import WikiText2, DatasetSplit
from model.language_model import MambaLM
from configs.config import MambaConfig
from utils.train_utils import *
from utils.test_utils import *
from utils.checkpoints import *


if __name__ == '__main__':
    parser = make_train_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = get_device_from_arg(args.device)

    dataset_args = dict(data_dir=args.data_dir, context_size=args.context_size, use_block_split=True)
    train_data = WikiText2(**dataset_args, split=DatasetSplit.train, max_vocab_size=args.max_vocab_size)
    valid_data = WikiText2(**dataset_args, split=DatasetSplit.valid, max_vocab_size=args.max_vocab_size)
    test_data = WikiText2(**dataset_args, split=DatasetSplit.test, max_vocab_size=args.max_vocab_size)

    # Setup model
    config = MambaConfig()
    config.vocab_size = len(train_data.word2idx)
    model = MambaLM(config,vocab_size=train_data.word_count(),device=device,)

    print("Restoring best checkpointed model...")
    parts = os.path.splitext(args.savefilename)
    best_checkpoint = f"{parts[0]}__best{parts[1]}"
    if os.path.isfile(best_checkpoint):
        print("Loading:", best_checkpoint)
        load_model_weights(model, best_checkpoint, device=device)
    else:
        print("Loading:", args.savefilename)
        load_model_weights(model, args.savefilename, device=device)

    test_metrics = evaluate_model(model, test_data, args.batch_size)
    print('=' * 89)
    print('| end of training | test loss {:5.2f} | test perplexity {:8.2f}'.format(test_metrics['loss'], test_metrics['ppl']))
    print(test_metrics)
    print('=' * 89)


    # print('\nUncurated samples')
    # print('-' * 89)
    # for i in range(5):
    #     print('({})'.format(i),sample_from_model(model, train_data))
