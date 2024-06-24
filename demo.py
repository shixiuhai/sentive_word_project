from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch

# 加载GPT-2的Tokenizer和Sequence Classification模型
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name)

# 定义敏感词汇
sensitive_words = ['敏感词1', '敏感词2', '色情','游行',"日逼","大获全胜","弑杀"]

# 将敏感词汇转换为token ID
sensitive_word_ids = tokenizer(sensitive_words, add_special_tokens=False)['input_ids']

def contains_sensitive_word(text):
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

# 测试例子
text_to_check = "那日逼宫，义军大获全胜，“军师”亲手弑杀暴君，一时间消息传出，举国上下尽是欢呼庆贺声。"
if contains_sensitive_word(text_to_check):
    print("文本包含违规词汇！")
else:
    print("文本没有包含违规词汇。")
