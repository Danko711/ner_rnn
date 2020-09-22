import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import gensim
import time
from catalyst import dl
from catalyst.contrib.nn.schedulers import OneCycleLRWithWarmup

from models import LstmCrf
from metrics import ner_token_f1
from data import Conll2003DatasetReader, ConllDataset
from conll_vectorizer import Vectorizer, PadSequence, Const

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

train_dl = DataLoader(data_train, batch_size=64, shuffle=True, collate_fn=PadSequence())
valid_dl = DataLoader(data_val, batch_size=64, shuffle=True, collate_fn=PadSequence())
dataloaders = {'train': train_dl, 'valid': valid_dl}

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
loss = model.loss

optimizer = optim.Adam(model.parameters())
scheduler = OneCycleLRWithWarmup(num_steps=4, lr_range=[7.5e-5, 1.5e-5, 1.0e-5], init_lr=3.0e-5, warmup_steps=1,
                                 optimizer=optimizer, decay_steps=1)

callbacks = {
    "optimizer": dl.OptimizerCallback(
        metric_key="loss",
        accumulation_steps=1,
        grad_clip_params=None
    ),
    "criterion": dl.CriterionCallback(
        input_key=['x', 'x_char', 'y'],
        output_key='pred'
    ),
    "metric": dl.MetricCallback(
        input_key='total_tags',
        output_key='preds',
        prefix='F1_token',
        metric_fn=ner_token_f1
    )
}

runner = dl.SupervisedRunner(
    input_key=['x', 'x_char'],
    output_key="preds",
    input_target_key="y",
)

runner.train(
    model=model,
    criterion=loss,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=dataloaders,
    callbacks=callbacks,
    logdir='./checkpoints',
    num_epochs=100,
    main_metric='F1_token',
    minimize_metric=False,
    verbose=True
)
