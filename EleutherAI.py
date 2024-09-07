from transformers import AutoTokenizer, AutoModelForCausalLM

# 选择一个模型，例如 GPT-J
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 编写你的提示
prompt = "你好，世界！"

# 编码提示并生成输出
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
