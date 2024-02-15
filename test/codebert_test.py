from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline

MODEL_DIR="/data/linux/transformers/microsoft/"  #microsoft/codebert-base-mlm
MODEL_NAME=MODEL_DIR+"codebert-base"
model = RobertaForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

CODE = "it is <mask> language: [ if ( a>5) { printf(\"hello!\");}]"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
outputs = fill_mask(CODE)
print(outputs)


print("*"*50)
CODE = "it is <mask> language: [ if a>5 print(\"hello\")]"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
outputs = fill_mask(CODE)
print(outputs)



# https://github.com/microsoft/CodeBERT/issues/53  source code classification

# https://github.com/NTDXYG/EL-CodeBert