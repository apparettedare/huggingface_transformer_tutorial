from transformers import AutoModel

checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
model = AutoModel.from_pretrained(checkpoint)