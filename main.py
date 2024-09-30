import pandas as pd
import os
import time
import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset


model_name = "Helsinki-NLP/opus-mt-en-mul"
# Set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the dataset from the CSV file
data_frame = pd.read_csv('sample.csv')

# Select the source and target languages
source_texts = data_frame['source'].tolist()
target_texts = data_frame['target_fr'].tolist()  # Use the desired target language

# Create a Dataset object
data = Dataset.from_dict({"source": source_texts, "target": target_texts})

# Load the model and tokenizer
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

model.to(device)  # Move the model to the specified device


# Define the preprocessing function
def preprocess_function(examples):
    # Tokenize the source texts and the target texts in a single call
    model_inputs = tokenizer(
        examples['source'], 
        text_target=examples['target'],  # Provide target texts here
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    # Move input tensors to the device
    for key in model_inputs:
        model_inputs[key] = model_inputs[key].to(device)

    return model_inputs

# Apply preprocessing to the dataset
tokenized_data = data.map(preprocess_function, batched=True)
train_test_split = tokenized_data.train_test_split(test_size=0.2, seed=42)  # Adjust seed for reproducibility
# Now you have train and validation datasets
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']


# Generate a unique run name based on the current timestamp
run_name = f"run-{int(time.time())}"  # Example: run-1632345678
logging_dir = os.path.join('./logs', run_name)

# Define training arguments with a learning rate scheduler
# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    save_steps=5000,
    save_total_limit=3,
    eval_strategy="steps",
    logging_strategy="steps",
    predict_with_generate=True,
    logging_dir=logging_dir,              # Directory for TensorBoard logs
    logging_steps=1,                 # Log every 500 steps
    eval_steps=1,                   # Evaluate every 1000 steps
    load_best_model_at_end=True,      # Load the best model at the end
    metric_for_best_model="eval_loss", # Metric for the best model
    greater_is_better=False,           # Lower loss is better
    fp16=True,                         # Enable mixed precision training
    gradient_accumulation_steps=2,     # Accumulate gradients
    learning_rate=2e-5,                # Learning rate
    lr_scheduler_type="linear",        # Learning rate scheduler
)

# Initialize the trainer with TensorBoard logging
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # You can provide a separate validation dataset here
)

# Fine-tune the model
trainer.train()

# To log gradients and weights histograms
def log_histograms(trainer):
    # This method will log the model's weights and gradients to TensorBoard
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            trainer.log({"weights/" + name: param, "gradients/" + name: param.grad})

# After training, you can call this function to log histograms
log_histograms(trainer)


