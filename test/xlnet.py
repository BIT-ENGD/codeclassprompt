from transformers import XLNetTokenizer, XLNetModel,XLNetLMHeadModel

tokenizer = XLNetTokenizer.from_pretrained('/data/linux/transformers/xlnet-large-cased')
model = XLNetLMHeadModel.from_pretrained('/data/linux/transformers/xlnet-large-cased')

inputs = tokenizer("Hello, my dog is cute <mask>", return_tensors="pt")
outputs = model(**inputs)

#last_hidden_states = outputs.last_hidden_state

#  class XLNetLMHeadModel
print(outputs)