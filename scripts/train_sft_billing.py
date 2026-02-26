"""
SFT training script for Milestone 6: Billing Category optimization.
Fine-tunes Qwen 1.5B on GPT-4 trajectories.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# MONKEYPATCH for transformers 5.1.0 compatibility with trl 0.8.6
# Must be applied BEFORE importing SFTTrainer
import transformers
import trl

print(f"DEBUG: transformers version: {transformers.__version__}")
print(f"DEBUG: trl version: {trl.__version__}")

orig_trainer_init = transformers.Trainer.__init__
def patched_trainer_init(self, *args, **kwargs):
    # Print what we're receiving to debug
    if "tokenizer" in kwargs:
        # print(f"DEBUG: Intercepted tokenizer in Trainer.__init__: {type(kwargs['tokenizer'])}")
        if "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            # If both are present, remove tokenizer as it's the one causing the error
            kwargs.pop("tokenizer")
    return orig_trainer_init(self, *args, **kwargs)

# Patch in both places to be sure
transformers.Trainer.__init__ = patched_trainer_init
if hasattr(transformers, "trainer"):
    transformers.trainer.Trainer.__init__ = patched_trainer_init

from trl import SFTTrainer
from src import config
from src.agents.tool_registry import get_tools_description

def train_sft():
    # 1. Load configuration
    model_name = config.RLConfig.MODEL_NAME
    data_path = "data/sft_billing_train.jsonl"
    output_dir = "models/qwen_sft_billing_final"
    
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Prepare prompt template (must match ReActLlamaModel)
    if config.Paths.RL_PROMPT_PATH.exists():
        system_prompt_template = config.Paths.RL_PROMPT_PATH.read_text()
    else:
        system_prompt_template = config.Paths.BASELINE_PROMPT_PATH.read_text()
    
    system_prompt = system_prompt_template.replace(
        "{TOOL_DESCRIPTIONS}", get_tools_description()
    )

    def formatting_func(example):
        # Construct the full text as the model would see it during inference
        # instruction: query
        # response: full ReAct trajectory string
        text = f"{system_prompt}\n\nQuestion: {example['instruction']}\n{example['response']}"
        return [text]

    # 3. Load dataset
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # 4. Training Arguments
    # User suggested: 5 epochs, LR=2e-5, batch_size=1, gradient_accumulation=4
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
    )

    # 5. Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 6. Run training
    print("Starting SFT training...")
    trainer.train()

    # 7. Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train_sft()
