import math
import numpy as np
import os
import pandas as pd
import time
import torch

from datasets import WikiText2, DatasetSplit
from MAMBA_barebones.datasets import WikiText2, DatasetSplit
from MAMBA_barebones.utils.train_utils import *
from MAMBA_barebones.utils.test_utils import *
from MAMBA_barebones.utils.checkpoints import *
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


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
    model = MambaLMHeadModel(config,vocab_size=train_data.word_count(),device=device,)

    print("Restoring best checkpointed model...")
    parts = os.path.splitext(args.savefilename)
    best_checkpoint = f"{parts[0]}__best{parts[1]}"
    if os.path.isfile(best_checkpoint):
        print("Loading:", best_checkpoint)
        load_model_weights(model, best_checkpoint, device=device)
    else:
        print("Loading:", args.savefilename)
        load_model_weights(model, args.savefilename, device=device)

    test_metrics = evaluate_model(model, test_data, args.batch_size,implementation='official')
    print('=' * 89)
    print('| end of training | test loss {:5.2f} | test perplexity {:8.2f}'.format(test_metrics['loss'], test_metrics['ppl']))
    print(test_metrics)
    print('=' * 89)


    # print('\nUncurated samples')
    # print('-' * 89)
    # for i in range(5):
    #     print('({})'.format(i),sample_from_model(model, train_data))
