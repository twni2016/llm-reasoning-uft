import datasets
import torch
import transformers
from trl import SFTTrainer
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F


class ULTrainer(SFTTrainer):
    def __init__(
        self,
        positive_dataset=None,
        negative_dataset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.positive_dataloader = self.get_ul_dataloader(positive_dataset)
        self.negative_dataloader = self.get_ul_dataloader(negative_dataset)
        self.positive_iter = iter(self.positive_dataloader)
        self.negative_iter = iter(self.negative_dataloader)
        assert 0.0 <= self.args.ul_alpha <= 0.1

    def get_ul_dataloader(self, dataset) -> DataLoader:
        data_collator = self.data_collator
        if transformers.utils.is_datasets_available() and isinstance(
            dataset, datasets.Dataset
        ):
            dataset = self._remove_unused_columns(dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                self.data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,  # the overall batch size is doubled
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = RandomSampler(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = transformers.trainer_utils.seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def training_step(self, model, inputs, num_items_in_batch=None):
        del inputs  # we don't use it
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        try:
            positive_batch = next(self.positive_iter)
        except StopIteration:
            self.positive_iter = iter(self.positive_dataloader)
            positive_batch = next(self.positive_iter)

        try:
            negative_batch = next(self.negative_iter)
        except StopIteration:
            self.negative_iter = iter(self.negative_dataloader)
            negative_batch = next(self.negative_iter)

        positive_batch = self._prepare_inputs(positive_batch)
        negative_batch = self._prepare_inputs(negative_batch)

        with self.compute_loss_context_manager():
            loss = self.compute_ul_loss(
                model,
                {"positive": positive_batch, "negative": negative_batch},
            )

        del positive_batch, negative_batch
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_ul_loss(self, model, raw_inputs, return_outputs=False):
        ## Note: forward twice is very cheap to compute

        pos_loss = self._standard_cross_entropy_loss(model, raw_inputs["positive"])

        if self.args.ul_loss_type == "gradient_ascent":
            neg_loss = -1.0 * self._standard_cross_entropy_loss(
                model, raw_inputs["negative"]
            )
        elif self.args.ul_loss_type == "unlikelihood":
            neg_loss = self._unlikelihood_loss(model, raw_inputs["negative"])
        else:
            raise NotImplementedError(f"unknown loss {self.args.ul_loss_type}")

        loss = (1.0 - self.args.ul_alpha) * pos_loss + self.args.ul_alpha * neg_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({"pos_loss": pos_loss.item(), "neg_loss": neg_loss.item()})

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, None) if return_outputs else loss

    def _standard_cross_entropy_loss(self, model, inputs):
        # min -log p(y | x)
        outputs = model(**inputs)

        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = outputs["logits"].float()
        labels = inputs["labels"]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        valid_counts = (shift_labels != -100).sum().float()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.sum() / valid_counts
        return loss

    def _unlikelihood_loss(self, model, inputs):
        # min -log (1-p(y | x))
        outputs = model(**inputs)

        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = outputs["logits"].float()
        labels = inputs["labels"]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        # create a mask
        valid_mask = shift_labels != -100
        valid_logits = shift_logits[valid_mask]
        valid_labels = shift_labels[valid_mask]

        # compute the loss
        probs = F.softmax(valid_logits, dim=-1)
        gold_probs = probs[torch.arange(valid_labels.shape[0]), valid_labels]
        ## -log(1-p(y|x))
        unlikelihood_loss = -torch.log(1 - gold_probs + 1e-8)
        loss = unlikelihood_loss.mean()

        return loss
