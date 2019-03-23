# Domain Adaptation of Word Vectors
### Assignment 2 - COL772(Spring'19): Natural Language Processing
#### Creator: Navreet Kaur[2015TT10917]
 
#### Problem Statement:
The goal of the assignment is to build a domain adaptation system for word vectors. The input of the code is a set of pre-trained word embeddings and a corpus of text, the domain of the textual given corpus is different from the domain of text that was used to train the word embeddings. The word embeddings are learnt given this data to show improvements for the task of masked language modelling.

#### The Task:
To write a model that constructs embeddings for the in-domain vocabulary. Also, given a sentence with one word masked out, the model should rank a given set of words in the order of how well the words fit the context in the place of the masked word. 
Note: The masked sentences would belong to the same domain as that of the training corpus.

#### Training Data:
All the data for required for this assignment is available in the following github repository:
```https://github.com/SaiKeshav/wv-da.git```

##### Input Files
1. ```eval_data.txt```: Every line contains a data-point which is a sentence with the masked word replaced with the following token: ```<<target>>``` , and the actual token is provided at the end of the line. In the dev file, The ground truth word is separated from the rest by a special delimiter ```::::``` (4 colons). The test_data file will have the same format but the actual word will be replaced by ‘dummy’.

 2.```eval_data.txt.td```: This file is line-aligned with the previous file. Every line contains a space-separated list of words, which form the target dictionary for each of the data-points in the previous file. Only one of the words is the correct answer but this set of words needs to be ranked and written in an output file with the respective ranks.

Link for pre-trained embeddings: ```https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit```

#### Instructions:
For downloading the trained model: ```sh compile.sh```

For training: ```sh train.sh input_data_folder_path model_path pretrained_embeddings_path evaluation_txt_file_path evaluation_txt_td_file_path```

For testing: ```sh test.sh eval_data.txt eval_data.txt.td model_path pretrained_embeddings_path```

This produces a file ```output.txt``` where each line contains the output for every data point in eval_data.txt. (Hence, it is line-aligned with eval_data.txt and eval_data.txt.td). Every line contains a space-separated list of ranks for each word mentioned in eval_data.txt.td (in the same order). The word with the highest probability (according to your model) is assigned a rank of 1.
