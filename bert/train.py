from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# 示例数据集
data = {
    'text': [
        "正常的文本内容",
        "违规的内容含有敏感词汇",
        "另一段正常的文本",
        "我们去一起去旅游好嘛",
        "参加游行示威是不对的",
        "游行示威不对的",
        "这种旅游行为很不好"
    ],
    'label': [0, 1, 0, 0, 1, 1, 0]  # 0表示正常，1表示违规
}

texts = np.array(data['text'])
labels = np.array(data['label'])

# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize the tokenizer with the desired maximum length
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
max_length = max(len(tokenizer.encode(text)) for text in texts)

# Prepare encodings
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=max_length)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Model initialization
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)  # binary classification

# Training arguments
training_args = TrainingArguments(
    output_dir='./output_dir',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_dir='./logs',
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Start training
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
model.save_pretrained('./bert-sensitivity-model')
