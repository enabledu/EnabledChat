from dataclasses import dataclass, field
from typing import Optional
import configparser
import os


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    input_column: str = field(
        metadata={
            "help": "The name of the column in the datasets containing the inputs."
        },
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the targets."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")

        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

def get_arguments():
    # requires modeification
    CURRENT_DIR = os.path.abspath(os.getcwd())
    TRAIN_DIR = CURRENT_DIR if CURRENT_DIR.split("/")[-1] == "train" else os.path.join(CURRENT_DIR, "train")
    CONFIG_DIR = os.path.join(TRAIN_DIR, "config.ini")
    
    config = configparser.ConfigParser()
    config.read(CONFIG_DIR)
    
    config_dict = {section: dict(config[section]) for section in config.sections()}

    for dic in config_dict.values():
        for key, value in dic.items():
            dic[key] = eval(value)
            
            if len(value.split(",")) != 1:
                dic[key] = value.strip('"').split(",")
            
    # model arguments
    model_args = config_dict["MODEL_ARGUMENTS"]
    
    # data training arguments
    data_args = config_dict["DATA_TRAINING_ARGUMENTS"]
    
    # training arguments
    training_args = config_dict["TRAINING_ARGUMENTS"]
    
    lora_config = config_dict["LORA_CONFIG"]
    
    return model_args, data_args, training_args, lora_config

if __name__=="__main__":
    ...