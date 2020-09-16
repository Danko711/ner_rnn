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

        seq = model(sents, chars, mask)
        seq_tens = [torch.Tensor(s) for s in seq]
        seq = torch.nn.utils.rnn.pad_sequence(seq_tens, batch_first=True).cpu().numpy()
        seq = torch.Tensor(seq)

        total_preds = [vectorizer.devectorize(i) for i in seq]
        total_tags = [vectorizer.devectorize(i) for i in tags]

        self.input = {'sents': sents, 'chars': chars, 'mask': mask, 'target': tags, 'total_tags': total_tags}
        self.output = {'preds': total_preds}



runner = CustomRunner()

runner.train(model=model,
             criterion=loss,
             optimizer=optimizer,
             loaders={'train': train_dl, 'valid': valid_dl},
             num_epochs=5,
             verbose=False,
             timeit=False,
             callbacks={
                 "optimizer": dl.OptimizerCallback(
                     metric_key="loss",
                     accumulation_steps=1,
                     grad_clip_params=None
                 ),
                 "criterion": dl.CriterionCallback(
                     input_key="total_tags",
                     output_key="preds"
                 ),
                 "metric": dl.MetricCallback(
                     input_key='total_tags',
                     output_key='preds',
                     prefix='F1_token',
                     metric_fn=ner_token_f1

                 )
             }
             )
