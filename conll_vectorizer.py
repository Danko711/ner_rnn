import torch


class Const:
    UNK_ID, UNK_TOKEN = -2, "<unk>"
    PAD_ID, PAD_TOKEN = -1, "<pad>"
    PAD_TAG_ID, PAD_TAG_TOKEN = -1, "<pad>"



class Vectorizer(object):
    base_word_to_ix = {
        Const.UNK_TOKEN: Const.UNK_ID,
        Const.PAD_TOKEN: Const.PAD_ID
    }

    base_ix_to_word = {
        Const.UNK_ID: Const.UNK_TOKEN,
        Const.PAD_ID: Const.PAD_TOKEN
    }

    tag_to_ix = {
        Const.PAD_TAG_TOKEN: Const.PAD_TAG_ID
    }

    def __init__(self, texts, tags):
        tokens = set([token for seq in texts for token in seq])
        self.word2Index = {word: index for index, word in enumerate(sorted(tokens))}
        self.index2Word = {index: word for index, word in enumerate(sorted(tokens))}
        self.word2Index = {**self.word2Index, **Vectorizer.base_word_to_ix}
        self.index2Word = {**self.index2Word, **Vectorizer.base_ix_to_word}

        char_tokens = set(list('АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'))
        self.char2Index = {str(char): index for index, char in enumerate(sorted(char_tokens))}
        self.index2Char = {index: str(char) for index, char in enumerate(sorted(char_tokens))}

        self.char2Index = {**self.char2Index, **Vectorizer.base_word_to_ix}
        self.index2Char = {**self.index2Char, **Vectorizer.base_ix_to_word}
        len_tags = len(Vectorizer.tag_to_ix)
        tags = set([tag for seq in tags for tag in seq])

        self.tags2Index = {tag: len_tags + index for index, tag in enumerate(sorted(tags))}
        self.tags2Index = {**self.tags2Index, **Vectorizer.tag_to_ix}
        self.index2tags = {index: tag for tag, index in self.tags2Index.items()}

    def lookup_index(self, token):
        if token in Vectorizer.base_word_to_ix:
            return Vectorizer.base_word_to_ix[token]
        else:
            return self.word2Index.get(token, Const.UNK_ID)

    def lookup_tag(self, tag):
        return self.tags2Index.get(tag, Const.UNK_ID)

    def size(self):
        return len(self.word2Index)

    def tag_size(self):
        return len(self.tags2Index)

    def char_size(self):
        return len(self.char2Index)

    def lookup_token(self, index):
        return self.index2Word.get(index, Const.UNK_TOKEN)

    def lookup_char(self, index):
        return self.char2Index.get(index, Const.UNK_ID)

    def vectorize(self, text, tags):
        return torch.LongTensor([self.lookup_index(token) for token in text]), \
               torch.LongTensor([self.lookup_char(char) for char in text]), \
               torch.LongTensor([self.lookup_tag(tag) for tag in tags])
    def devectorize(self, tag_idx):
        return [self.index2tags[int(id)] for id in tag_idx]



class PadSequence(object):
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        char_sequences = [x[1] for x in sorted_batch]
        char_sequences_padded = torch.nn.utils.rnn.pad_sequence(char_sequences, batch_first=True)

        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = [x[2] for x in sorted_batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return sequences_padded, char_sequences_padded, lengths, labels_padded
