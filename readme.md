# Weakly Supervised Learning for Textbook Question Answering

## Introduction
Textbook Question Answering (TQA) is the task of answering diagram and non-diagram questions given large multi-modal contexts consisting of a lot of text and diagrams. Deep text understandings and effective learning of diagram semantics are important for this task due to its specificity. In this paper, we propose a Weakly Supervised learning method for TQA (WSTQ), which regards the incompletely accurate results of essential intermediate procedures for TQA as supervision to develop Text Matching (TM) and Relation Detection (RD) tasks and then employs the tasks to motivate itself to learn strong text comprehension and excellent diagram semantics respectively. Specifically, we apply the result of text retrieval to build positive as well as negative TM pairs. In order to learn deep text understandings, we first pre-train the text understanding module of WSTQ on TM and then fine-tune it on TQA. We build positive as well as negative RD pairs by checking whether there is any overlap between the items/regions detected from diagrams using object detection. The RD task is used to force our method to learn the relationships between regions, which are crucial to express the diagram semantics. We train WSTQ on RD and TQA simultaneously, \emph{i.e.}, multitask learning, to obtain effective diagram semantics and then improve the TQA performance. Extensive experiments are carried out on CK12-QA and AI2D to verify the effectiveness of WSTQ. Experimental results show that our method achieves significant accuracy improvements of $5.02\%$ and $4.12\%$ on test splits of the above datasets respectively than the current state-of-the-art baseline. 
The details of this paper can be seen at .
<div style="align: center">
<img https://github.com/dr-majie/WSTQ/blob/master/framework.png width=60% />
</div>
<img src=https://github.com/dr-majie/WSTQ/blob/master/framework.png width=60% align="center"/>

## File description
Here we only upload the pre-processed text and you should download the images in https://allenai.org/data/tqa
```
.
|-- framework.png           # our framework
|-- jsons                   # pre-processed CK12-QA dataset
|   |-- tqa_dmc.json        # DMC questions 
|   |-- tqa_ndmc.json       # NDMC questions
|   `-- tqa_ndtf.json       # NDTF questions
|-- models                  
|   |-- tqa_dmc.py          # DMC model
|   |-- tqa_ndmc.py         # NDMC model
|   |-- tqa_ndtf.py         # NDTF model
|   `-- txt_matching.py     # Pre-training a Text Matching (TM) model
|-- readme.md
|-- run_dmc.py              # main function to train DMC model
|-- run_ndmc.py             # main function to train NDMC model
|-- run_ndtf.py             # main function to train NDTF model
|-- test.py                 # main funciton to test model
`-- utils
    |-- engine.py           # train and test a specific model
    `-- tools.py            # some functions called by public
```

## Install dependencies
```
tokenizers      0.10.3
pytorch         1.5.0 
transformers    4.11.3      
```

## How to run
```
python run_dmc.py/run_ndmc.py/run_ndtf.py to train a specific model
python test.py to test a specific model
```
