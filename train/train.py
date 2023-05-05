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
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
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

    # load the model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        train_args.gradient_accumulation_steps = (
            train_args.gradient_accumulation_steps // world_size
        )

    if model_args.use_fast_tokenizer:
        tokenizer: PreTrainedTokenizer = LlamaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=model_args.tokenizer_name,
            add_eos_token=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.tokenizer_name,
            add_eos_token=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        load_in_8bit=True,
        device_map=device_map,
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

    trainer.train()

    # Push model to hub
    if model_args.push_model_to_hub and train_args.hub_token is not None:
        model.push_to_hub(train_args.hub_model_id)
        tokenizer.push_to_hub(train_args.hub_model_id)


if __name__ == "__main__":
    main()
