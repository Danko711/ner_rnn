import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import gensim
import time
from catalyst import dl

from models import LstmCrf
from metrics import ner_token_f1
from data import Conll2003DatasetReader, ConllDataset
from conll_vectorizer import Vectorizer, PadSequence, Const

reader = Conll2003DatasetReader()
data = reader.read(dataset_name='conll2003', data_path='./')

texts = pd.Series([i[0] for i in data['train']])
tags = pd.Series([i[1] for i in data['train']])

vectorizer = Vectorizer(texts=texts, tags=tags)

data_train = ConllDataset(data, 'train', vectorizer)
data_val = ConllDataset(data, 'valid', vectorizer)
print("Dataset ready")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ft_vectors = gensim.models.fasttext.load_facebook_model('./fasttext/fasttext/wiki.simple.bin')
print('fasttext loaded')

train_dl = DataLoader(data_train, batch_size=64, shuffle=True, collate_fn=PadSequence())
valid_dl = DataLoader(data_val, batch_size=64, shuffle=True, collate_fn=PadSequence())

model = LstmCrf(ft_vectors.wv.vectors,
                vectorizer.size(),
                vectorizer.tag_size(),
                embedding_dim=300,
                hidden_dim=128,
                char_vocab_size=vectorizer.char_size(),
                char_embedding_dim=64,
                char_in_channels=128,
                device=device)

model.to(device)

loss = model.loss

optimizer = optim.Adam(model.parameters())


class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        sents, chars, lengths, tags = batch

        self.model.train()

        sents, chars, tags = sents.to(device), chars.to(device), tags.to(device)
        mask = (tags != Const.PAD_TAG_ID).float()
        mask.to(device)

        loss = self.model.loss(sents, chars, tags, mask)

        seq = model(sents, chars, mask)
        seq_tens = [torch.Tensor(s) for s in seq]
        seq = torch.nn.utils.rnn.pad_sequence(seq_tens, batch_first=True).cpu().numpy()
        seq = torch.Tensor(seq)

        f1 = ner_token_f1(tags, seq)

        self.batch_metrics.update({"loss": loss, "F1": f1})

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        if self.is_train_loader:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


runner = CustomRunner()

runner.train(model=model,
             optimizer=optimizer,
             loaders={'train': train_dl, 'valid': valid_dl},
             num_epochs=3,
             verbose=True,
             timeit=True
             )
