# Transformer N-Gram Markov Representations

## Installation

To install and run locally, run the following command from within this directory:

```{bash}
pip install -e .
```

To install for development, first install [Rye](https://rye-up.com/) via the instructions on their site. Then run:

```{bash}
rye sync
```
To install and build all of the local dependencies. Then you can `source .venv/bin/activate` to enter the development virtual environment.


## Make TinyStories512 dataset and n-gram index

Activate the virtual environment and run:

```{bash}
python3 scripts/preprocess_data.py \
    --dataset "roneneldan/TinyStories" \
    --dataset-key "train" \
    --tokenizer-path tokenizer/tinystories512/ \
    --output-prefix data/tinystories512 \
    --workers 8 \
    --append-bos
```

From there you can build the n-gram index if you have built `tokengrams`. From within python:

```{python}
from tokengrams import MemmapIndex
index = MemmapIndex.build(
    'data/tinystories512/tinystories512_train_document.bin',
    'data/tinystories_ngrams.idx',
    verbose=True
)
```

To calculate the bigrams:

```{python}
queries = [[i] for i in range(512)]
max_index = 511
# Returns a 512x512 list of bigram counts, i.e. bigram_counts[i, j] is the number of times in the corpus that token j appears directly after token i
bigram_counts = index.batch_count_next(queries, max_index)
```
