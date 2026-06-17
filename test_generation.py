import os
import torch
from datasets import WikiText2, DatasetSplit
from model.language_model import MambaLM
from configs.config import MambaConfig
from utils.train_utils import *
from utils.test_utils import *
from utils.checkpoints import *
import random


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
    config = MambaConfig(d_model=args.d_model,n_layer=args.n_layer,d_state=args.d_state,dropout=args.dropout,)
    config.vocab_size = len(train_data.word2idx)
    model = MambaLM(config,device=device,)

    print("Restoring best checkpointed model...")
    #parts = os.path.splitext(args.savefilename)
    #best_checkpoint = f"{parts[0]}__best{parts[1]}"
    best_checkpoint = "experiments/logs/0/checkpoints/exp_021__best.pt"
    if os.path.isfile(best_checkpoint):
        print("Loading:", best_checkpoint)
        load_model_weights(model, best_checkpoint, device=device)
    else:
        print("Loading:", args.savefilename)
        load_model_weights(model, args.savefilename, device=device)

    model.eval()

    #prompt_words = ["The","son","of"]
    #prompt_ids = [train_data.word2idx[w]for w in prompt_words]
    #prompt = torch.tensor([prompt_ids],device=device)
    idx2word = {v: k for k, v in train_data.word2idx.items()}
    idx = random.randint(0, len(test_data) - 1)
    x, y = test_data[idx]
    prompt = x[:10].unsqueeze(0).to(device)
    generated = model.generate(prompt=prompt,max_length=50,top_k=40, top_p=0.95, temperature=0.8,min_p = 0.1,penalty = 1.1,
                               eos_token_id=train_data.word2idx["<eos>"])
    tokens = generated[0].cpu().tolist()
    prompt_words = [idx2word[t] for t in prompt[0].cpu().tolist()]
    print("PROMPT:")
    print(" ".join(prompt_words))
    words = [idx2word.get(token, "<UNK>")for token in tokens]
    print("OUTPUT:")
    print(" ".join(words)) 
    print("Generated shape:", generated.shape)
    print('=' * 89)
