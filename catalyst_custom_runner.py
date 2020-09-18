import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import gensim
import time
from catalyst import dl
from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup

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

dataloaders = {'train': train_dl, 'valid': valid_dl}

loss = model.loss

optimizer = optim.Adam(model.parameters())
scheduler = OneCycleLRWithWarmup(optimizer=optimizer, num_steps=4, lr_range=[7.5e-5, 1.5e-5, 1.0e-5], init_lr=3.0e-5,
                                 warmup_steps=1, decay_steps=1)


class CustomRunner(dl.Runner):


    def _handle_batch(self, batch):
        sents, chars, lengths, tags = batch

        self.model.train()

        sents, chars, tags = sents.to(device), chars.to(device), tags.to(device)
        #mask = (tags != Const.PAD_TAG_ID).float()
        #mask.to(device)

        seq = model(sents, chars)#, mask)
        seq_tens = [torch.Tensor(s) for s in seq]
        seq = torch.nn.utils.rnn.pad_sequence(seq_tens, batch_first=True).cpu().numpy()
        seq = torch.Tensor(seq)


        if seq.size()[1] != tags.size()[1]:
            for i, j in zip(seq_tens, tags):
                print(i.size(), j.size())


        total_preds = [vectorizer.devectorize(i) for i in seq]
        total_tags = [vectorizer.devectorize(i) for i in tags]

        if len(total_tags) != len(total_preds):
            print('tags: ', len(total_tags))
            print('preds', len(total_preds))

        self.input = {'x': sents, 'x_char': chars, 'y': tags, 'total_tags': total_tags} #'mask': mask,
        self.output = {'preds': total_preds}



runner = CustomRunner()

runner.train(model=model,
             criterion=loss,
             optimizer=optimizer,
             loaders=dataloaders,
             num_epochs=50,
             verbose=False,
             timeit=False,
             scheduler=None,
             callbacks={
                 "optimizer": dl.OptimizerCallback(
                     metric_key="loss",
                     accumulation_steps=1,
                     grad_clip_params=None
                 ),
                 "criterion": dl.CriterionCallback(
                     input_key=['x', 'x_char', 'y'], #'mask': mask,
                     output_key=[]
                 ),
                 "metric": dl.MetricCallback(
                     input_key='total_tags',
                     output_key='preds',
                     prefix='F1_token',
                     metric_fn=ner_token_f1


                 )
             }
             )
