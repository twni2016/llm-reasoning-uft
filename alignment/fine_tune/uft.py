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


import logging
import os, sys
import datasets
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    set_seed,
    DataCollatorForLanguageModeling,
)
from trl import DataCollatorForCompletionOnlyLM

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    UFTConfig,
    get_checkpoint,
    get_datasets,
    trim_negative_data,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from fine_tune.data_prep import template_API
from fine_tune.uft_trainer import UFTTrainer


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, UFTConfig))
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
    assert training_args.packing is False  # to avoid contamination across examples

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
    logger.info(f"UFTConfig: {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load and preprocess datasets
    ###############
    positive_dataset = get_datasets(
        data_args.positive_dataset_sources,
        columns_to_keep=[
            training_args.dataset_text_field,
        ],
    )
    negative_dataset = get_datasets(
        data_args.negative_dataset_sources,
        columns_to_keep=[
            training_args.dataset_text_field,
        ],
    )

    template = template_API(data_args.dataset_root)
    positive_dataset = positive_dataset.map(
        lambda example: template(example, training_args.dataset_text_field),
        num_proc=training_args.dataset_num_proc,
        desc="templating positive dataset",
    )
    negative_dataset = negative_dataset.map(
        lambda example: template(example, training_args.dataset_text_field),
        num_proc=training_args.dataset_num_proc,
        desc="templating negative dataset",
    )

    # this saves time for tokenization and training
    negative_dataset = trim_negative_data(
        positive_dataset,
        negative_dataset,
        training_args.dataset_text_field,
        num_proc=training_args.dataset_num_proc,
    )
    ## finally select the subset of negative data we need
    # assume 1:1 in batch split; save time for tokenization
    # we have shuffled the data before, so the order does not matter
    negative_data_size = min(
        len(positive_dataset) * int(training_args.num_train_epochs),
        len(negative_dataset),
    )
    logger.info(
        f"reducing negative data size from {len(negative_dataset)} to {negative_data_size}"
    )
    negative_dataset = negative_dataset.select(range(negative_data_size))

    for dataset, desc in zip(
        [positive_dataset, negative_dataset], ["positive", "negative"]
    ):
        logger.info(f"Training on the {desc} dataset: {dataset}")
        logger.info(f"{dataset.to_pandas()['source'].value_counts(normalize=True)}")

        with training_args.main_process_first():
            for index in [0, 1, 2, 3, 4]:
                logger.info(
                    f"Sample {desc} index {index} \n{dataset[index][training_args.dataset_text_field]}"
                )

    #####################
    # Tokenization
    #####################
    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples[training_args.dataset_text_field],
            truncation=True,
            max_length=training_args.max_seq_length,
        )

    positive_dataset = positive_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=training_args.dataset_num_proc,
        remove_columns=[training_args.dataset_text_field],
        desc="tokenizing positive dataset",
    )
    negative_dataset = negative_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=training_args.dataset_num_proc,
        remove_columns=[training_args.dataset_text_field],
        desc="tokenizing negative dataset",
    )

    if data_args.train_on_completion_only:
        response_template_ids = tokenizer.encode("\nSteps:\n", add_special_tokens=False)
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids, tokenizer=tokenizer
        )
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
    trainer = UFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=positive_dataset,  # dummy for logging
        positive_dataset=positive_dataset,  # actually used for training
        negative_dataset=negative_dataset,
        data_collator=collator,
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
