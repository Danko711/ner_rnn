import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import gensim
import time
from catalyst import dl

from models import LstmCrf
from data import Conll2003DatasetReader, ConllDataset
from conll_vectorizer import Vectorizer, PadSequence


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

ft_vectors = gensim.models.fasttext.load_facebook_model('./fasttext/wiki.simple.bin')
print('fasttext loaded')


train_dl = DataLoader(data_train, batch_size=64, shuffle=True, collate_fn=PadSequence())
test_dl = DataLoader(data_val, batch_size=64, shuffle=True, collate_fn=PadSequence())



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

loss = model.loss()

optimizer = optim.Adam(model.parameters())


runner = dl.SupervisedRunner()

runner.train(
    loaders={"train": train_dl, "valid": test_dl},
    model=model, criterion=loss, optimizer=optimizer,
    num_epochs=1, logdir="./logs", verbose=True,
    load_best_on_end=True,
)
