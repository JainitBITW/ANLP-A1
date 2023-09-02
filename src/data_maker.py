from pprint import pprint
import re 
import nltk 
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import pickle
import argparse
import torch 
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from pprint import pprint
import re 
import config_hp as hp 
import os 

class Custom_Dataset(Dataset):
    '''This class helps in creating the Pretrain Dataset'''
    def __init__(self, data):
        random.shuffle(data)
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx]['sentence']),torch.tensor( self.data[idx]['label']))
    

def custom_collate(batch):
    '''Custom collate function for pretraining task'''
    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences to the maximum length in the batch
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return ( padded_sentences,padded_labels)


def parse_args():
    parser = argparse.ArgumentParser(description='Take args for data')
    parser.add_argument('--input', type=str, default='/home2/jainit/ANLP-A1/data/Auguste_Maquet.txt', help='input file name')
    parser.add_argument('--ngram', action="store_true", help="Set this flag to make the argument True")
    parser.add_argument('--n', type=int, default=5, help='ngram')
    parser.add_argument('--output', type=str, default='data', help='output folder name')
    parser.add_argument('--glove_path', type=str, default='/home2/jainit/ANLP-A1/glove/glove.6B.50d.txt', help='random seed')
    args = parser.parse_args()
    return args


def preprocess(corpus):
    '''
    preprocess the corpus
    args:  corpus: str
    return: corpus: str
    '''
    corpus = re.sub(r'\n', ' ', corpus)
    corpus = re.sub(r' +', ' ', corpus)
    corpus = re.sub(r'\.', '.', corpus)
    corpus = re.sub(r'™', ' <TM> ', corpus)
    corpus = re.sub(r'’|’|‘', '\'', corpus)
    corpus = re.sub(r'“|”|‘|\`\`|\'\' ', '\"', corpus)
    corpus = re.sub(r',', ',', corpus)
    corpus = re.sub(r'•', ' ', corpus)
    corpus = re.sub(r'–|—|-', ' ', corpus)
    corpus = re.sub(r'_', ' ', corpus)
    corpus = re.sub(r'\d+', '<NUM>', corpus)
    # replace a link with <LINK>
    corpus = re.sub(r'http\S+', '<LINK>', corpus)
    return corpus



def tokenize(corpus ,glove,train_split=0.5 , val_split=0.16, test_split=0.34 ):
    '''
    tokenize the corpus
    args:  corpus: str, split: float , glove: dict
    return: words_all: list[list[str]]
    '''
    # tokenize the corpus
    embedding_dim = len(list(glove.values())[0])
    sentences = sent_tokenize(corpus)
    train_sentences = sentences[:int(len(sentences)*train_split)]
    val_sentences = sentences[int(len(sentences)*train_split):int(len(sentences)*(train_split+val_split))]
    test_sentences = sentences[int(len(sentences)*(train_split+val_split)):]
    words_all = {'train':[], 'val':[], 'test':[]}
    splits = {'train':train_sentences, 'val':val_sentences, 'test':test_sentences}
    word_count = {}
    unknown_words = set()
    known_words = set()
    for split in splits:
        for sentence in splits[split]:
            words = word_tokenize(sentence)
            words = [word.lower() for word in words]
            words = ['<SOS>'] + words + ['<EOS>']
            # if split == 'train': # so that we dont bias with test set 
            for i in range(len(words)):
                if words[i] not in word_count:
                    word_count[words[i]] = 1
                else:
                    word_count[words[i]] += 1
            words_all[split].append(words)
    
    for split in words_all:
        for i , sentences in enumerate (words_all[split]):
            for j, word in enumerate(sentences):
                if word not in word_count or word_count[word] < hp.THRESHOLD :
                    words_all[split][i][j] = '<OOV>'
                    unknown_words.add(word)
                else:
                    known_words.add(word)
    known_words.add('<PAD>')
    known_words.add('<SOS>')
    known_words.add('<EOS>')
    known_words.add('<OOV>')
    known_words.add('<NUM>')
    known_words.add('<LINK>')

    embeddings = np.zeros((len(known_words), len(list(glove.values())[0])))
    words_to_idx = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<OOV>':3, '<NUM>':4, '<LINK>':5}
    for i, word in enumerate(known_words):
        if word not in words_to_idx:
            words_to_idx[word] = len(words_to_idx)
        if word not in glove:
            glove[word] = torch.rand(embedding_dim)
        embeddings[words_to_idx[word]] = glove[word]
    embeddings = torch.FloatTensor(embeddings)
    print(f'Number of unknown words: {len(unknown_words)}')
    print(f'Number of known words: {len(known_words)}') 
    for split in words_all:
        for i , sentences in enumerate (words_all[split]):
            for j, word in enumerate(sentences):
                if word not in words_to_idx:
                    words_all[split][i][j] = words_to_idx['<OOV>']
                else:
                    words_all[split][i][j] = words_to_idx[word]
    return sentences, words_all, embeddings, words_to_idx , list(known_words)


def make_glove_dict(path):
    glove = {}
    embedding_dim= 0 
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            glove[line[0]] = torch.tensor([float(x) for x in line[1:]])
            embedding_dim = len(line[1:])

    glove['<OOV>'] = torch.mean(torch.stack(list(glove.values())), dim=0)
    glove['<PAD>'] = torch.zeros(embedding_dim)
    glove['<SOS>'] = torch.rand(embedding_dim)
    glove['<EOS>'] = torch.rand(embedding_dim)
    glove['<NUM>'] = torch.rand(embedding_dim)
    glove['<LINK>'] = torch.rand(embedding_dim)
    return glove



def make_dataloaders(data, ngram=True):
    datasets = {'train':Custom_Dataset(data['train']), 'val':Custom_Dataset(data['val']), 'test':Custom_Dataset(data['test'])}
    if ngram:
        dataloaders = {'train':DataLoader(datasets['train'], batch_size=hp.BATCH_SIZE, shuffle=True), 'val':DataLoader(datasets['val'], batch_size=hp.BATCH_SIZE, shuffle=True), 'test':DataLoader(datasets['test'], batch_size=hp.BATCH_SIZE, shuffle=True)} 
    else:
        dataloaders = {'train':DataLoader(datasets['train'], batch_size=hp.BATCH_SIZE ,collate_fn = custom_collate, shuffle=True), 'val':DataLoader(datasets['val'], batch_size=hp.BATCH_SIZE,collate_fn = custom_collate, shuffle=True), 'test':DataLoader(datasets['test'], batch_size=hp.BATCH_SIZE,collate_fn = custom_collate, shuffle=True)} 

    return dataloaders

if __name__ == '__main__': 
    args = parse_args()
    glove = make_glove_dict(args.glove_path)
    # read file 
    with open(args.input, 'r') as f:
        text = f.read()
    print(args)
    # preprocess the text
    text = preprocess(text)
    # tokenize the text
    sentences, words_all , embeddings,word_to_id  , vocab  = tokenize(text, glove)
    with open(f'../{args.output}/vocab.json', 'w') as f:
        json.dump(vocab, f)
    with open(f'../{args.output}/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    with open(f'../{args.output}/word_to_id.json', 'w') as f:
        json.dump(word_to_id, f)
    if args.ngram:
        print(f'Creating ngrams with n = {args.n}' )
        n = args.n
        n+=1
        ngrams = {'train':[], 'val':[], 'test':[]}

        for split in words_all:
            for sentence in words_all[split]:
                for i in range(len(sentence)-n+1):
                    ngrams[split].append(sentence[i:i+n])
        
        ngrams_data = {'train':[], 'val':[], 'test':[]}

        for split in ngrams:
            for ngram in ngrams[split]:
                ngrams_data[split].append({'sentence':ngram[:-1], 'label':ngram[-1]})

        dataloaders = make_dataloaders(ngrams_data)
        # create data folder 
        if not os.path.exists('data'):
            os.makedirs('data')

        with open(f'../{args.output}/ngrams_dataloader.pkl', 'wb') as f:
            pickle.dump(dataloaders, f)
            
        with open(f'../{args.output}/ngrams_data.json', 'w') as f:
            json.dump(ngrams_data, f)
    else: 
        print('Creating the language model dataset')
        data = {'train':[], 'val':[], 'test':[]}
        for split in words_all:
            for sentence in words_all[split]:
                data[split].append({'sentence':sentence[:-1], 'label':sentence[1:]})
        dataloaders = make_dataloaders(data, ngram=False)
        # create data folder
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(f'../{args.output}/lm_dataloader.pkl', 'wb') as f:
            pickle.dump(dataloaders, f)
        with open(f'../{args.output}/lm_data.json', 'w') as f:
            json.dump(data, f)
        
        
        
    
        
    
