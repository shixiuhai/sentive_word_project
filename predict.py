from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
import torch

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

# 定义敏感词汇
sensitive_words = ['敏感词1', '敏感词2', '色情', '游行',"日逼","弑杀"]

# 将敏感词汇转换为token ID
sensitive_word_ids = tokenizer(sensitive_words, add_special_tokens=False)['input_ids']

def contains_sensitive_word(text, model, tokenizer):
    # 使用GPT-2 Tokenizer编码文本
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # 使用GPT-2 Sequence Classification模型进行分类
    outputs = model(input_ids)
    logits = outputs.logits
    
    # 获取分类结果（这里假设模型输出的第一个类别对应无违规，第二个类别对应有违规）
    predicted_class = torch.argmax(logits, dim=-1)
    
    # 判断是否有违规词汇
    if predicted_class == 1:
        return True
    else:
        return False

# 加载微调后的模型和tokenizer
def load_model_and_tokenizer(model_path):
    model = GPT2ForSequenceClassification.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer
# 测试例子
def main():
    model_path = "./output_dir"
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    text_to_check = "那日逼宫，义军大获全胜，“军师”亲手弑杀暴君，一时间消息传出，举国上下尽是欢呼庆贺声。"
    if contains_sensitive_word(text_to_check, model, tokenizer):
        print("文本包含违规词汇！")
    else:
        print("文本没有包含违规词汇。")

if __name__ == "__main__":
    main()
