# Steps to run the code 

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
python3 data_maker.py  --input <file.txt> --ngram --n 5 --output <dir_name>

``` 
