import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load cleaned dataset
df = pd.read_csv("archive (2)/cleaned_fake_news.csv")

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize function
def tokenize(batch):
    titles = [str(t) for t in batch["title"]]  # ensure clean list of strings
    print("Batch titles sample:", titles[:3])
    return tokenizer(titles, padding="max_length", truncation=True, max_length=128)

    

# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./bert-fake-news")
tokenizer.save_pretrained("./bert-fake-news")

print("✅ Model trained and saved to ./bert-fake-news")
