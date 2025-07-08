#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

## Based on
# CPO: Contrastive Preference Optimization https://arxiv.org/abs/2401.08417
# SimPO: Simple Preference Optimization with a Reference-Free Reward https://arxiv.org/abs/2405.14734


import logging
import os, sys
import datasets
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    set_seed,
)

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    PrefConfig,
    get_checkpoint,
    get_datasets,
    trim_negative_data,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from fine_tune.data_prep import template_API
from datasets import Dataset
from collections import defaultdict
import random
from trl import CPOTrainer
from tqdm import tqdm


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, PrefConfig))
    model_args, data_args, training_args = parser.parse()

    model_name = model_args.model_name_or_path.split("/")[-1]
    data_args.positive_dataset_sources = [
        os.path.join(data_args.dataset_root, s.replace("<model_name>", model_name))
        for s in data_args.positive_dataset_sources
    ]
    data_args.negative_dataset_sources = [
        os.path.join(data_args.dataset_root, s.replace("<model_name>", model_name))
        for s in data_args.negative_dataset_sources
    ]

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Config: {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    positive_dataset = get_datasets(
        data_args.positive_dataset_sources,
        columns_to_keep=[
            "text",
        ],
    )
    negative_dataset = get_datasets(
        data_args.negative_dataset_sources,
        columns_to_keep=[
            "text",
        ],
    )

    template = template_API(data_args.dataset_root)
    positive_dataset = positive_dataset.map(
        lambda example: template(example, "text"),
        num_proc=training_args.dataset_num_proc,
        desc="templating positive dataset",
    )
    negative_dataset = negative_dataset.map(
        lambda example: template(example, "text"),
        num_proc=training_args.dataset_num_proc,
        desc="templating negative dataset",
    )
    # this saves time for tokenization and training
    negative_dataset = trim_negative_data(
        positive_dataset,
        negative_dataset,
        "text",
        num_proc=training_args.dataset_num_proc,
    )

    ###############
    # Preprocess datasets for preference-based methods
    ###############
    # bucket rows by case_id
    pos_by_id = defaultdict(list)
    for row in positive_dataset:
        pos_by_id[row["case_id"]].append(row["text"])

    neg_by_id = defaultdict(list)
    for row in negative_dataset:
        neg_by_id[row["case_id"]].append(row["text"])

    shared_ids = set(pos_by_id) & set(neg_by_id)  # only ids with both pos & neg

    # build preference rows
    DELIM = "\nSteps:\n"
    N_pairs = int(training_args.num_copy_positive)
    pref_rows = []
    num_neg_per_pos = []

    for cid in tqdm(shared_ids, desc="Building preference rows"):
        pos_texts = pos_by_id[cid]
        neg_texts = neg_by_id[cid]

        # ---- extract prompt (assumes identical across all examples)
        prompt = pos_texts[0].split(DELIM, 1)[0] + DELIM

        # ---- split out completions once to avoid repeating work
        pos_completions = [t.split(DELIM, 1)[1] for t in pos_texts]
        neg_completions = [t.split(DELIM, 1)[1] for t in neg_texts]
        num_neg_per_pos.append(
            len(neg_completions) if len(neg_completions) < N_pairs else N_pairs
        )

        for chosen in pos_completions:
            # ---- for each positive, create <= N preference pairs
            if len(neg_completions) >= N_pairs:
                # enough uniques → sample without replacement
                rejected_pool = random.sample(neg_completions, k=N_pairs)
            else:
                # not enough uniques → sample with replacement
                rejected_pool = random.choices(neg_completions, k=N_pairs)

            for rejected in rejected_pool:
                pref_rows.append(
                    {
                        "case_id": cid,
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

    pref_dataset = Dataset.from_list(pref_rows).shuffle(seed=training_args.seed)

    ## stats
    logger.info(f"{len(shared_ids) / len(pos_by_id) = }")
    logger.info(
        f"avg num of negative per positive: {sum(num_neg_per_pos) / len(num_neg_per_pos)}"
    )
    logger.info(f"{len(pref_dataset) / (len(positive_dataset) * N_pairs) = }")
    logger.info(f"{len(positive_dataset) = }")

    logger.info(f"Training on the preference dataset: {pref_dataset}")
    with training_args.main_process_first():
        for index in [0, 1, 2, 3, 4]:
            logger.info(f"Sample index {index} \n{pref_dataset[index]}\n")

    #####################
    # Tokenization
    #####################
    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.pad_token = tokenizer.eos_token

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    ########################
    # Initialize the Trainer
    ########################
    trainer = CPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=pref_dataset,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_state()
    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
