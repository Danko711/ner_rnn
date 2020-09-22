import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import gensim
import time

from models import LstmCrf
from data import Conll2003DatasetReader, ConllDataset
from conll_vectorizer import Vectorizer, PadSequence
from train_eval import train, evaluate


reader = Conll2003DatasetReader()
data = reader.read(dataset_name='conll2003', data_path='./')

texts = pd.Series([i[0] for i in data['train']])
tags = pd.Series([i[1] for i in data['train']])

print('start loading fasttext')
ft_vectors = gensim.models.fasttext.load_facebook_model('./fasttext/fasttext/wiki.simple.bin')
print('Fasttext loaded')
vectorizer = Vectorizer(texts=texts, tags=tags, word_embedder=ft_vectors)
print('vectorizer ready')

data_train = ConllDataset(data, 'train', vectorizer)
data_val = ConllDataset(data, 'valid', vectorizer)
print("Dataset ready")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import os
print(os.getcwd())


train_dl = DataLoader(data_train, batch_size=64, shuffle=True, collate_fn=PadSequence())
test_dl = DataLoader(data_val, batch_size=64, shuffle=True, collate_fn=PadSequence())


model = LstmCrf(vectorizer.embedding_matrix,
                vectorizer.size(),
                vectorizer.tag_size(),
                embedding_dim=300,
                hidden_dim=128,
                char_vocab_size=vectorizer.char_size(),
                char_embedding_dim=64,
                char_in_channels=128,
                device=device)

model.to(device)

optimizer = optim.Adam(model.parameters())


n_epochs = 5

clip = 1
best_test_loss = float('inf')

for epoch in range(n_epochs):
    print("EPOCH ", epoch, " START #########################################")
    start_time = time.time()

    train_loss = train(model, train_dl, optimizer, clip)
    test_loss, f1 = evaluate(model, test_dl, vectorizer)

    end_time = time.time()

    epoch_time = end_time - start_time

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model, f'./checkpoints/ner_lstm_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }, f'./checkpoints/ner_bilstm_crf_epoch_{epoch}.pt')

    print(f'\nEpoch {epoch + 1} | Time {epoch_time}')
    print(f'\nTrain loss  {train_loss}')
    print(f'\nF1 score {f1}')
    print(f'\nTest loss {test_loss}')
