import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import (
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from utils import DataTrainingArguments, LoraTrainingConfig, ModelArguments


def main():
    model_args: ModelArguments
    data_args: DataTrainingArguments
    train_args: TrainingArguments
    lora_config: LoraTrainingConfig

    parser = HfArgumentParser(
        dataclass_types=(
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            LoraTrainingConfig,
        )
    )

    if len(sys.argv) == 2:  # passing a json file
        model_args, data_args, train_args, lora_config = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            train_args,
            lora_config,
        ) = parser.parse_args_into_dataclasses()

    # Login in to HuggingFace Hub
    if train_args.hub_token is not None:
        login(train_args.hub_token)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir) and not train_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and train_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the project name for wandb
    if model_args.wandb_project_name is not None:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project_name

    # load the dataset
    dataset = load_dataset(
        path=data_args.dataset_name,
        name=data_args.dataset_config_name,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    dataset["train"] = (
        dataset["train"].select(range(data_args.max_train_samples))
        if data_args.max_train_samples
        else dataset["train"]
    )

    dataset["validation"] = (
        dataset["validation"].select(range(data_args.max_eval_samples))
        if data_args.max_eval_samples
        else dataset["validation"]
    )

    # Load model and tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.tokenizer_name,
        add_eos_token=True,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        train_args.gradient_accumulation_steps = (
            train_args.gradient_accumulation_steps // world_size
        )

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=torch.float16,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = prepare_model_for_int8_training(model)

    model: PeftModel = get_peft_model(model, lora_config)
    tokenizer.pad_token_id = 0

    def tokenize(data):
        result = tokenizer(
            data[data_args.input_column],
            truncation=True,
            max_length=data_args.max_source_length,
            padding="max_length" if data_args.pad_to_max_length else None,
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        num_proc=data_args.preprocessing_num_workers,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=train_args,
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("\n", "*" * 50)
    model.print_trainable_parameters()
    print("*" * 50, end="\n\n")

    checkpoint = None
    if train_args.resume_from_checkpoint is not None:
        checkpoint = train_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # Push final model to hub
    if train_args.push_to_hub:
        model.push_to_hub(train_args.hub_model_id)
        tokenizer.push_to_hub(train_args.hub_model_id)


if __name__ == "__main__":
    main()
