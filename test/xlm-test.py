from transformers import pipeline
unmasker = pipeline('fill-mask', model='/data/linux/transformers/xlm-roberta-large')
out=unmasker("Hello I'm a <mask> model.")
print(out)