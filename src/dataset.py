import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from collections import defaultdict
import random
from tqdm import tqdm

class TaoBaoDataset(Dataset):
    def __init__(self, data_dir, max_len=15,n_rows=80000000, split='train', test_ratio=0.2, random_seed=42, mask_prob=0.15):
        self.data_dir = data_dir
        self.n_rows=n_rows
        self.max_len = max_len
        self.split = split
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.mask_prob = mask_prob
        self.processed_file = os.path.join(data_dir, f'processed_data_{max_len}_{test_ratio}_{random_seed}.pkl')

        if os.path.exists(self.processed_file):
            self.load_processed_data()
        else:
            self.process_and_save_data()

        self.rng = random.Random(self.random_seed)

    def process_and_save_data(self):
        taobao_file= os.path.join(self.data_dir, 'UserBehavior.csv')
        column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
        print('Reading Data: ',end='')
        # 80M was the highest number of row that can be loaded
        self.df=pd.read_csv(taobao_file,nrows=self.n_rows,header=None,names=column_names)
        self.df=self.df[self.df['behavior_type']=='buy']
        print('Done!')
        print('Mapping items to index: ',end='')
        self.item_id_to_idx={id: idx + 1 for idx, id in enumerate(self.df['item_id'].unique())}
        self.df['item_id'] = self.df['item_id'].map(self.item_id_to_idx)
        print('Done!')
        self.num_items=len(self.item_id_to_idx) + 1 # +1 for padding
        self.mask_token=self.num_items + 1 # Last index is for the mask token

        print('Build User Sequence: ',end='')
        # Build the user interaction sequence
        self.df=self.df.sort_values(['user_id', 'timestamp'])
        user_sequences=self.df.groupby('user_id').apply(
            lambda x: x.sort_values('timestamp')['item_id'].tolist()).to_dict()
        print('Done!')

        print('Calculate item popularity: ',end='')
        # Calculating item popularity (for negative sampling) 
        self.item_popularity = self.df['item_id'].value_counts().to_dict()
        print('Done!')

        # Divide the data into train\val\test
        train_sequences = []
        val_sequences = []
        val_targets = []
        test_sequences = []
        test_targets = []
        negative_samples = {}

        np.random.seed(self.random_seed)
        for user_id, sequence in tqdm(user_sequences.items()):
            if len(sequence) > 2:  # A minimum of 3 items is required to divide it
                if len(sequence) > self.max_len:
                    sequence = sequence[-self.max_len:]  # Pre-truncation

                train_seq = sequence[:-2]
                val_seq = sequence[:-1]
                test_seq = sequence

                train_sequences.append((user_id, train_seq))
                val_sequences.append((user_id,val_seq))
                val_targets.append(val_seq[-1])
                test_sequences.append((user_id,test_seq))
                test_targets.append(test_seq[-1])

                # Generate 100 negative samples for each user
                neg_samples = self.sample_negative_items(sequence[-1], 100)
                negative_samples[user_id] = neg_samples

        # Save the processed data
        with open(self.processed_file, 'wb') as f:
            pickle.dump((train_sequences, val_sequences, val_targets, test_sequences, test_targets, negative_samples, self.item_id_to_idx, self.item_popularity, self.num_items, self.mask_token), f)
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.val_targets = val_targets
        self.test_sequences = test_sequences
        self.test_targets = test_targets
        self.negative_samples = negative_samples
        self.set_split(self.split)

    def load_processed_data(self):
        with open(self.processed_file, 'rb') as f:
            train_sequences, val_sequences, val_targets,\
            test_sequences, test_targets, negative_samples, \
            self.item_id_to_idx, self.item_popularity, self.num_items, self.mask_token = pickle.load(f)


        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.val_targets = val_targets
        self.test_sequences = test_sequences
        self.test_targets = test_targets
        self.negative_samples = negative_samples
        self.set_split(self.split)

    def set_split(self, split):
        self.split = split
        if split == 'train':
            self.sequences = self.train_sequences
        elif split == 'val':
            self.sequences = self.val_sequences
            self.targets = self.val_targets
        elif split == 'test':
            self.sequences = self.test_sequences
            self.targets = self.test_targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.split == 'train':
            user, seq = self.sequences[idx]
            tokens = []
            labels = []
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

            return torch.LongTensor(tokens), torch.LongTensor(labels)
        else:
            user_id,seq = self.sequences[idx]
            answer = [self.targets[idx]]
            negs = self.negative_samples[user_id]

            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)

            seq = seq + [self.mask_token]
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

    def sample_negative_items(self, positive_item, num_samples):
        # store item_ids and item_probs
        item_ids = np.array(list(self.item_popularity.keys()))
        item_probs = np.array(list(self.item_popularity.values()), dtype=float)
        item_probs /= item_probs.sum()

        # Create a mask that excludes positive samples
        mask = item_ids != positive_item
        item_ids_filtered = item_ids[mask]
        item_probs_filtered = item_probs[mask]
        item_probs_filtered /= item_probs_filtered.sum()  # Renormalize probability

        # All negative samples are sampled at once and then deduplicated
        samples = np.random.choice(
            item_ids_filtered,
            size=num_samples * 2,  # Sample a little more to prevent duplicates
            replace=False,
            p=item_probs_filtered
        )

        # Deduplicate and truncate to the desired quantity
        negative_items = list(dict.fromkeys(samples))[:num_samples]

        return negative_items
