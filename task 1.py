from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Step 1: Load the Pre-trained GPT-2 Model and Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Explicitly set the padding token as the eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Step 2: Load and Preprocess the Dataset using the datasets library
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

# Load your custom dataset
dataset = load_dataset('text', data_files={'train': 'path_to_your_train_dataset.txt'})  # Replace with your dataset path
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Step 3: Set up the Data Collator (helps in dynamic padding)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Step 5: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

# Step 6: Fine-tune the Model
trainer.train()

# Step 7: Save the Model
trainer.save_model("./fine_tuned_gpt2")

# Step 8: Generate Text Using the Fine-tuned Model with attention_mask
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Explicitly set the padding token as the eos_token again after reloading the model
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

prompt = "Your custom prompt here"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,  # Controls randomness
    top_k=50,  # Only sample from the top k tokens
    top_p=0.9,  # Nucleus sampling
    repetition_penalty=1.2,  # Penalize repetition
    do_sample=True  # Enable sampling for diverse generation
)


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:\n", generated_text)