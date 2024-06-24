import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型和分词器
model_path = './bert-sensitivity-model'  # 这里是你保存模型的路径
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(model_path)
def predict_sensitivity(text, sensitive_words, model, tokenizer):
    results = {}
    for word in sensitive_words:
        # 构建输入文本和敏感词汇的组合句子
        combined_text = f"[CLS] {text} [SEP] {word} [SEP]"
        
        # 使用分词器编码输入文本
        encoded_inputs = tokenizer(combined_text, padding=True, truncation=True, return_tensors='pt')

        # 推断模式下进行预测
        with torch.no_grad():
            outputs = model(**encoded_inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

        # 提取敏感词汇的预测概率
        results[word] = probabilities[:, 1].item()  # 取出违规（1）类别的概率

    return results


# 示例文本和敏感词汇
input_text = "参加游行示威是不对的"
sensitive_words = ["敏感词汇", "违规词汇", "不当","游行"]

# 进行预测
predictions = predict_sensitivity(input_text, sensitive_words,model,tokenizer)

# 打印预测结果
for word, prob in predictions.items():
    print(f"敏感词汇 '{word}' 的预测概率为: {prob:.4f}")
    
    
    
    
    
    
    
    
    
    
    
# def predict_sensitivity(text, sensitive_words):
#     # 将文本和敏感词汇组合成句子，例如："[CLS] 这是一段包含敏感词汇的文本内容 [SEP] 敏感词汇 [SEP]"
#     inputs = []
#     for word in sensitive_words:
#         inputs.append('[CLS] ' + text + ' [SEP] ' + word + ' [SEP]')
    
#     # 使用分词器编码输入文本
#     encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')

#     # 推断模式下进行预测
#     with torch.no_grad():
#         outputs = model(**encoded_inputs)

#     logits = outputs.logits
#     probabilities = torch.softmax(logits, dim=1)

#     # 返回敏感词汇的预测概率
#     results = {sensitive_words[i]: probabilities[i, 1].item() for i in range(len(sensitive_words))}
#     return results
