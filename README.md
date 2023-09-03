# Steps to run the code 
 
## File Structure

```bash
.
├── Adv_NLP_Assignment_1.pdf
├── data
│   ├── Auguste_Maquet.txt
│   ├── embeddings.pkl
│   ├── lm_data.json
│   ├── lm_dataloader.pkl
│   ├── model_10.pth
│   ├── models
│   │   ├── LSTM.pt
│   │   └── NNLM.pt
│   ├── ngrams_data.json
│   ├── ngrams_dataloader.pkl
│   ├── sentences.json
│   ├── vocab.json
│   ├── vocab.pkl
│   └── word_to_id.json
├── glove
│   ├── glove.6B.100d.txt
│   ├── glove.6B.200d.txt
│   ├── glove.6B.300d.txt
│   └── glove.6B.50d.txt
├── perplexities
│   ├── 2021114003-LM1-test-perplexity.txt
│   ├── 2021114003-LM1-train-perplexity.txt
│   ├── 2021114003-LM1-val-perplexity.txt
│   ├── 2021114003-LM2-test-perplexity.txt
│   ├── 2021114003-LM2-train-perplexity.txt
│   └── 2021114003-LM2-val-perplexity.txt
├── __pycache__
│   ├── config.cpython-39.pyc
│   └── config_hp.cpython-39.pyc
├── README.md
└── src
    ├── config_hp.py
    ├── data
    ├── data_maker.py
    ├── __init__.py
    ├── LSTM.py
    ├── model_rerun.py
    ├── NNLM.py
    ├── __pycache__
    │   ├── config_hp.cpython-39.pyc
    │   ├── data_maker.cpython-39.pyc
    │   ├── LSTM.cpython-39.pyc
    │   └── NNLM.cpython-39.pyc
    ├── refs
    │   ├── code.ipynb
    │   └── lstm.ipynb
    ├── rerun.ipynb
    
```
Make sure you have the same file structure as above

### 1. Install the requirements 
    
```bash
pip3 install -r requirements.txt
```

### 2.  Download the glove embeddings 

```bash
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ..
```


# Part 1 -> NNLM 

### 3.  Make the Ngram Data using file [data_maker.py](data_maker.py)
    
```bash
python data_maker.py  --input <file.txt> --ngram --n 5 --glove <glove_file>

``` 
### 4.  Train the NNLM model using file [nnlm.py](nnlm.py)
    
```bash
python NNLM.py  --save --wandb

```
to use wandb logging use --wandb flag
to save the model use --save flag

### 5.  Test the saved NNLM model using file [model_rerun.py](model_rerun.py)
There are two watys to test the model
1. Using the test data 
```bash
python model_rerun.py  --model nnlm --test --model_file <model_file>

```
2. Using the input sentence
```bash

python model_rerun.py  --model nnlm --model_file <model_file>

```

# Part 2 -> LSTM

### 3. Make the Ngram Data using file [data_maker.py](data_maker.py)
    
```bash
python data_maker.py  --input <file.txt>  --glove <glove_file>

```
### 4. Train the LSTM model using file [LSTM.py](LSTM.py)
    
```bash

python LSTM.py  --save --wandb

```
to use wandb logging use --wandb flag
to save the model use --save flag

### 5. Test the saved LSTM model using file [model_rerun.py](model_rerun.py)
There are two watys to test the model
1. Using the test data 
```bash
python model_rerun.py  --model lstm --test --model_file <model_file>

```
2. Using the input sentence
```bash

python model_rerun.py  --model lstm --model_file <model_file>

```
File link for full dataset
https://iiitaphyd-my.sharepoint.com/:u:/g/personal/jainit_bafna_research_iiit_ac_in/EUsDRztt_eJKic_rsmj3P7EBVre6h-6WboRsA3dBAcPIHw?e=IrXfNn




