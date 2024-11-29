from transformers import BertModel

checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
model = BertModel.from_pretrained(checkpoint)
model.save_pretrained("./tmp")