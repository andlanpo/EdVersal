from transformers import LlamaForCausalLM, LlamaTokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("Llama-model-identifier")
model = LlamaForCausalLM.from_pretrained("Llama-model-identifier")

# Prepare dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path_to_train_file.json",
    block_size=128
)

valid_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path_to_validation_file.json",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./Llama_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=500,
    warmup_steps=500,
    prediction_loss_only=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./Llama_finetuned")
