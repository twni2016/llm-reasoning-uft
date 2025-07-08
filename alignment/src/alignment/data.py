# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_datasets(
    dataset_sources: List,
    columns_to_keep: Optional[List[str]] = None,
) -> DatasetDict:
    """
    Args:
        dataset_sources (`List`):
            Dictionary containing the dataset names.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
    Returns
        [`Dataset`]: The dataset containing the loaded datasets.
    """
    raw_datasets = []

    for ds in dataset_sources:
        # assure each *individual* dataset has been deduplicated
        dataset = load_dataset("csv", data_files={"train": ds})["train"]
        dataset = dataset.add_column(
            "source", ["/".join(ds.split("/")[-2:])] * len(dataset)
        )
        raw_datasets.append(dataset)

    raw_datasets = concatenate_datasets(raw_datasets).shuffle(seed=42)

    # deduplicate (duplication is caused by concatenation of datasets)
    processed_datasets = Dataset.from_pandas(
        raw_datasets.to_pandas().drop_duplicates(subset=columns_to_keep),
        preserve_index=False,
    )
    return processed_datasets


def trim_negative_data(
    positive_dataset,
    negative_dataset,
    text_field: str,
    num_proc: int,
):

    # First trim the lengths
    def calculate_length(example):
        example["text_length"] = len(example[text_field])
        return example

    def trim_text(example, max_length):
        example[text_field] = example[text_field][:max_length]
        return example

    dataset_with_length = positive_dataset.map(
        calculate_length,
        num_proc=num_proc,
        desc="calculating lengths in positive dataset",
    )

    # use the max length of positive dataset as cap
    max_length = max(dataset_with_length["text_length"])
    negative_dataset = negative_dataset.map(
        lambda example: trim_text(example, max_length),
        num_proc=num_proc,
        desc="trimming lengths in negative dataset",
    )

    ## Then deduplicate due to the shortened lengths
    negative_dataset = Dataset.from_pandas(
        negative_dataset.to_pandas().drop_duplicates(subset=text_field),
        preserve_index=False,
    )

    return negative_dataset
