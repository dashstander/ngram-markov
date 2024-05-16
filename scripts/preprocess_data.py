# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)


import time
import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer
from ngram_markov.indexed_dataset import make_builder


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)

    def encode(self, text):
        
        ids = {}
        key = self.args.dataset_key
        doc_ids = []
        text_ids = Encoder.tokenizer.encode(text[key])
        if self.args.append_bos:
            doc_ids.append(Encoder.tokenizer.bos_token_id)
        if len(text_ids) > 0:
            doc_ids.append(text_ids)
        if self.args.append_eod:
            doc_ids[-1].append(Encoder.tokenizer.bos_token_id)
        ids[key] = doc_ids
        return ids, len(text)


def get_args(input_args=None):
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--dataset-key",
        default="train",
        help="space separate listed of keys to extract from dataset. Default: train",
    )

    group = parser.add_argument_group(title="tokenizer")

    group.add_argument('--tokenizer-path', type=str, required=True)

    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument(
        "--append-bos",
        action="store_true",
        help="Append a <bos> token to the end of a document.",
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
   

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args(input_args)
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_dataset(dataset, key):
    for example in dataset[key]:
        yield {key: example['text']}


def main(input_args=None):
    args = get_args(input_args)
    encoder = Encoder(args)
    dataset = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    key = args.dataset_key

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # use multiprocessing to iterate over input documents
    fin = yield_from_dataset(dataset, key)

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    output_bin_files[key] = "{}_{}_{}.bin".format(
        args.output_prefix, key, "document"
    )
    output_idx_files[key] = "{}_{}_{}.idx".format(
        args.output_prefix, key, "document"
    )
    builders[key] = make_builder(
        output_bin_files[key],
        impl='mmap',
        vocab_size=tokenizer.vocab_size,
    )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            #print(sentences)
            for sentence in sentences:
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
            # separate with eos token
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i} documents ({i / elapsed :.2f} docs/s, {mbs:.2f} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
