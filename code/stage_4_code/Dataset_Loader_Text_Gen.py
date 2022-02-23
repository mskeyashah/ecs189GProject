'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
from collections import Counter
import torch

class Dataset_Loader_Text_Gen(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    sequence_length = 4
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv('../../data/stage_4_data/text_generation/data')
        text = train_df['Joke'].str.cat(sep=' ')
        return text.split(' ')


    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)


    def __len__(self):
        return len(self.words_indexes) - self.sequence_length


    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.sequence_length]),
            torch.tensor(self.words_indexes[index + 1:index + self.sequence_length + 1]),
        )