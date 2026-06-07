import math
import textwrap
import torch

@torch.no_grad()
def evaluate_model(model, data, batch_size, topk=(1, 3, 5, 10)):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    device = next(model.parameters()).device
    loss = 0.
    correct = torch.zeros(len(topk), dtype=torch.long)
    total = 0
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
    for i, (x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss += criterion(logits.reshape(-1, logits.size(-1)),y.reshape(-1)).item()
        # compute accuracies
        targets = y.reshape(-1)
        mask = torch.ones_like(targets, dtype=torch.bool)
        for idx in data.non_vocab_idx:
            mask &= (targets != idx)
        predictions = logits.reshape(-1, logits.size(-1))
        correct += _get_topk_correct(targets[mask],predictions[mask],topk)
        total += mask.double().sum().item()
    model.train()
    loss = loss / len(loader)
    accuracies = {f'accuracy_top{k}': correct[i].item()/total for i, k in enumerate(topk)}
    return dict(loss=loss, ppl=math.exp(loss), **accuracies)

def _get_topk_correct(y, scores, topk):
    # y: (B,), yhat: (B, n_classes)
    y_pred = scores.topk(k=max(topk), dim=1)[1].t()  # (B, K_max) -> (K_max, B)
    y1 = y.view(1, -1).expand_as(y_pred)  # (K_max, B); each column is identical
    correct = (y_pred == y1)  # (K_max, B); which predictions are correct
    return torch.LongTensor([correct[:k].sum().item() for k in topk])

def sample_from_model(model, train_data):
    words = []
    model.eval()
    device = next(model.parameters()).device
    history = torch.randint(train_data.word_count(), (1, 1), dtype=torch.long).to(device)
    context_size = train_data.context_size
    for i in range(context_size):
        logits = model(history)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        word_idx = torch.multinomial(probs.squeeze(0), 1).item()
        word_tensor = torch.Tensor([[word_idx]]).long().to(device)
        history = torch.cat([history, word_tensor],dim=1)
        words.append(train_data.idx2word[word_idx])

    return '\n'.join(textwrap.wrap(' '.join(words),80))
