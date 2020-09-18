from tqdm import tqdm
import torch

from itertools import chain

from conll_vectorizer import Const
from metrics import ner_token_f1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, iterator, optimizer, clip):
    model.train()
    epoch_loss = 0
    for i, (sents, chars, lengths, tags) in tqdm(enumerate(iterator), total=len(iterator)):
        sents, chars, tags = sents.to(device), chars.to(device), tags.to(device)
        #mask = (tags != Const.PAD_TAG_ID).float()
        #mask.to(device)
        optimizer.zero_grad()

        loss = model.loss(sents, chars, tags)#, mask)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, vectorizer):
    model.eval()
    epoch_loss = 0
    total_preds = []
    total_tags = []
    with torch.no_grad():
        for i, (sents, chars, lengths, tags) in tqdm(enumerate(iterator), total=len(iterator)):
            sents, chars, tags = sents.to(device), chars.to(device), tags.to(device)
            #mask = (tags != Const.PAD_TAG_ID).float()
            #mask.to(device)
            loss = model.loss(sents, chars, tags)#, mask)

            seq = model(sents, chars)#, mask)
            seq_tens = [torch.Tensor(s) for s in seq]
            seq = torch.nn.utils.rnn.pad_sequence(seq_tens, batch_first=True).cpu().numpy()
            seq = torch.Tensor(seq)

            total_preds += [vectorizer.devectorize(i) for i in seq]
            total_tags += [vectorizer.devectorize(i) for i in tags]

            epoch_loss += loss.item()

    print('tags: ', len(chain(*total_tags)))
    print('preds: ', len(chain(*total_preds)))

    f1 = ner_token_f1(total_tags, total_preds)
    return epoch_loss / len(iterator), f1
