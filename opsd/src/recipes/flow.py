"""Flow-matching SFT recipe independent of the model's attention pattern."""

import torch

from dataset import LatentSequenceDataset, validate_training_dataset

from .base import (
    add_common_training_arguments,
    per_sample_mse,
    validate_common_training_arguments,
)


class FlowRecipe:
    name = "action2v"
    progress_description = "Training"

    def add_arguments(self, parser, backend):
        if not backend.supports_sft:
            raise TypeError(f"{type(backend).__name__} does not implement SFT")
        backend.add_training_arguments(parser)
        add_common_training_arguments(parser, "video-sft")
        parser.set_defaults(
            max_train_steps=150,
            save_every=50,
            lora_rank=0,
            lora_alpha=256,
            learning_rate=5.0e-6,
        )

    def validate_arguments(self, parser, args):
        validate_common_training_arguments(parser, args)

    def model_config(self, backend, args):
        return backend.model_config(args)

    def dataset(self, args):
        return LatentSequenceDataset(args.data_path, load_latent=True)

    def validate_dataset(self, backend, dataset, config):
        backend.validate_model_config(config)
        validate_training_dataset(dataset, backend)

    def training_config(self, args, metadata):
        return {
            "trainer": self.name,
            "num_samples": args.num_samples,
            "data": metadata,
            "few_step": False,
        }

    def schedule(self, backend, device, config, args):
        return backend.schedule(device, config)

    def reset_metrics(self, device):
        self.loss_sum = torch.zeros((), device=device)

    def accumulate(self, per_sample_loss):
        self.loss_sum += per_sample_loss.detach().sum()

    def pop_metric_sums(self):
        values = {"train/loss": self.loss_sum.detach().clone()}
        self.loss_sum.zero_()
        return values

    def forward(
        self, backend, model, batch, context, teacher_context,
        device, schedule, config,
    ):
        output = backend.flow_forward(
            model,
            batch,
            context,
            device,
            schedule,
            config=config,
            secondary_context=teacher_context,
        )
        return per_sample_mse(output)
