import os
import sys
import pickle
import random
import json

import torch
import torch.nn as nn
import bitsandbytes as bnb

from datasets import load_dataset

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from utils import (
    ModelArguments,
    DataTrainingArguments,
    get_arguments
)



if __name__=="__main__":
    
    model_args, data_args, training_args, lora_config = get_arguments()
    
    # load the dataset
    dataset = load_dataset(
        path=data_args["dataset_name"],
        name=data_args["dataset_config_name"],
        use_auth_token=True if model_args["use_auth_token"] else None,
    )
    
    # load the model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    gradient_accumlation_steps = training_args["batch_size"] // training_args["per_device_train_batch_size"]
    
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumlation_steps = gradient_accumlation_steps // world_size
    
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args["tokenizer_name"],
        add_eos_token=True,
        use_auth_token=True if model_args["use_auth_token"] else None,
    )
    
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args["model_name"],
        load_in_8bit=True,
        device_map=device_map,
        use_auth_token=True if model_args["use_auth_token"] else None,
    )
    
    model = prepare_model_for_int8_training(model)
    
    config = LoraConfig(
        r=lora_config["lora_r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0
    
    def tokenize(data):
        result = tokenizer(
            data[data_args["input_column"]],
            truncation=True,
            max_length=data_args["max_source_length"],
            padding="max_length" if data_args["pad_to_max_length"] else None,
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }
    
    tokenized_dataset = dataset.map(
        tokenize, 
        batched=True, 
        remove_columns=dataset["validation"].column_names,
        batch_size=data_args["preprocessing_num_workers"]
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
        args=TrainingArguments(
            per_device_train_batch_size=training_args["per_device_train_batch_size"],
            per_device_eval_batch_size=training_args["per_device_eval_batch_size"],
            gradient_accumulation_steps=gradient_accumlation_steps,
            warmup_steps=training_args["warmup_steps"],
            num_train_epochs=training_args["num_train_epochs"],
            learning_rate=training_args["learning_rate"],
            fp16=training_args["fp16"],
            logging_steps=training_args["logging_steps"],
            evaluation_strategy=training_args["evaluation_strategy"],
            save_strategy=training_args["save_strategy"],
            eval_steps=training_args["eval_steps"],
            save_steps=training_args["save_steps"],
            output_dir=training_args["output_dir"],
            save_total_limit=training_args["save_total_limit"],
            load_best_model_at_end=training_args["load_best_model_at_end"],
            ddp_find_unused_parameters=False if ddp else None,
        ),
    )
    
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()