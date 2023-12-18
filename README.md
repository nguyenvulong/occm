# occm
My attempt to employ one-class learning to detect synthetic speech.
In fancy terms, I am trying to transform a hyper-plane classifier into a sphere covering positive samples.

For feature extraction (frontend), I use wav2vec model from Meta and finetune it with a subset of real/synthetic samples.
For classification (backend), I work with several models SE-Resnet, AASIST, etc. 
This work is under progress. All suggestions are welcome.

# Note

## install this version of fairseq
`pip install git+https://github.com/facebookresearch/fairseq.git@a54021305d6b3c4c5959ac9395135f63202db8f1`

## fix numpy package if necessary
float -> float64
