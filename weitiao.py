from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# 加载预训练的GPT-2模型和Tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 添加一个特殊的 padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 修正模型初始化时的配置，确保传递给模型的padding token正确配置
model = GPT2ForSequenceClassification.from_pretrained(
    model_name,
    pad_token_id=tokenizer.pad_token_id  # 传递tokenizer的padding token id
)

# 定义微调的数据集类
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }
        return item


# 定义微调的数据集（示例）
train_texts = [
    "这是一段包含游行的文本。",
    "这是一段关于旅游的文本。",
    # 添加更多的训练文本例子，包括不同语义的句子
]
train_labels = [1, 0]  # 1 表示包含敏感词汇（游行），0 表示不包含敏感词汇（旅游）

# 初始化自定义的数据集对象
train_dataset = MyDataset(train_texts, train_labels, tokenizer)

# 定义Trainer的训练参数
training_args = TrainingArguments(
    output_dir='./output_dir',  # 指定输出目录，用于保存日志和模型检查点
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
)

# 定义Trainer进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 开始微调
trainer.train()
# 在训练完成后手动保存模型
# 保存模型和分词器
model.save_pretrained('./output_dir')
tokenizer.save_pretrained('./output_dir')