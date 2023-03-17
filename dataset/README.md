<center><h3>5g text classification</h3></center>

<h4>How to run:</h4>
1. Firstly, we need to install all the packages we need


      pip install -r requirements.txt

2. Then open the terminal and enter the following file path:
/intern/

    input the following command in the terminal:


    python main.py --model={model_name}

    

There are three models for you to choose:
   1. TextCNN
   2. TextRNN
   3. Transformer

So here is an example of the command:

    python main.py --model=TextCNN


3. The model will use pretrained parameters by default, if you don't want to use the pretrained model, you can add:


    --embedding=random

Here is an example:

    python main.py --model=TextCNN --embedding=random


<h4>Structure of the Project</h4>
1. dataset:

   This directory is used to store the dataset and logs
2. models:
   
   This directory is used to store the models

3. dataloader.py
   
   This python file is used to preprocess the dataset 
and then build the processed dataset used to train and test

4. main.py

   This python file is used to start the project.

5. train_eval.py

   This python file defines how the models train and test, all the functions 
   related to training, testing are here


<h4>My thoughts about the project</h4>
Basically, the project is not that hard even though I failed to implement in the afternoon.

And the structure I design make the code more readable and explicit. 
And it could could me to build other projects for other applications.

The process of the project is as follow:
1. open .csv file and get the content and category
2. Building the vocabulary list (corpus: including the UNK and PAD) and transform all the words into index(the number that the word in the corpus)
3. dividing the dataset into three parts: Training set, Validation set and Test set.
4. building iterators for each dataset in convenience of training and testing.
5. Config all the parameters for model, dataset and training
6. init model and then used to train
7. evaluate the model
