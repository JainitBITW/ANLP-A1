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

class Ngram_Dataset(Dataset):
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
    parser.add_argument('--input', type=str, default='Auguste_Maquet.txt', help='input file name')
    parser.add_argument('--ngram', action="store_true", help="Set this flag to make the argument True")
    parser.add_argument('--n', type=int, default=5, help='ngram')
    parser.add_argument('--output', type=str, default='data', help='output folder name')
    parser.add_argument('--glove_path', type=str, default='glove/glove.6B.50d.txt', help='random seed')
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



def tokenize(corpus ,vocab,train_split=0.5 , val_split=0.16, test_split=0.34 ):
    '''
    tokenize the corpus
    args:  corpus: str, split: float , glove: dict
    return: words_all: list[list[str]]
    '''
    # tokenize the corpus
    sentences = sent_tokenize(corpus)
    train_sentences = sentences[:int(len(sentences)*train_split)]
    val_sentences = sentences[int(len(sentences)*train_split):int(len(sentences)*(train_split+val_split))]
    test_sentences = sentences[int(len(sentences)*(train_split+val_split)):]
    words_all = {'train':[], 'val':[], 'test':[]}
    splits = {'train':train_sentences, 'val':val_sentences, 'test':test_sentences}
    word_count = {}
    unknwon_words = set()
    known_words = set()
    for split in splits:
        for sentence in splits[split]:
            words = word_tokenize(sentence)
            words = [word.lower() for word in words]
            words = ['<SOS>'] + words + ['<EOS>']
            
            for i in range(len(words)):
                if words[i] not in word_count:
                    word_count[words[i]] = 1
                else:
                    word_count[words[i]] += 1
                if words[i] not in vocab:
                    
                    unknwon_words.add(words[i])
                    words[i] = vocab['<OOV>']
                    
                else:
                    known_words.add(words[i])
                    words[i] = vocab[words[i]]
            words_all[split].append(words)
    print('Tokenization done')
    print(f'Number of unknown words: {len(unknwon_words)}')
    print(f'Number of known words: {len(known_words)}')
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    with open('wordcount.txt', 'w') as f:
        f.write(str(word_count))
    return sentences, words_all

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

def make_vocab(glove):
    vocab = {}
    embeddings = {}
    for word in glove:
        vocab[word] = len(vocab)
        embeddings[vocab[word]] = glove[word]
    return vocab, embeddings

def make_dataloaders(data, ngram=True):
    if ngram:
        datasets = {'train':Ngram_Dataset(data['train']), 'val':Ngram_Dataset(data['val']), 'test':Ngram_Dataset(data['test'])}
        dataloaders = {'train':DataLoader(datasets['train'], batch_size=hp.BATCH_SIZE, shuffle=True), 'val':DataLoader(datasets['val'], batch_size=hp.BATCH_SIZE, shuffle=True), 'test':DataLoader(datasets['test'], batch_size=hp.BATCH_SIZE, shuffle=True)} 
    else:
        return None
    return dataloaders

if __name__ == '__main__': 
    args = parse_args()
    glove = make_glove_dict(args.glove_path)
    vocab,embeddings= make_vocab(glove)
    # create pkl for embeddings and vobac 
    with open(f'{args.output}/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(f'{args.output}/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    # read file 
    with open(args.input, 'r') as f:
        text = f.read()
    print(args)
    # preprocess the text
    text = preprocess(text)
    # tokenize the text
    sentences, words_all = tokenize(text, vocab)
    
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

        with open(f'{args.output}/ngrams_dataloader.pkl', 'wb') as f:
            pickle.dump(dataloaders, f)
            
        with open(f'{args.output}/ngrams_data.json', 'w') as f:
            json.dump(ngrams_data, f)
        
        
    
        
    
