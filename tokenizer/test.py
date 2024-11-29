from transformers import BertJapaneseTokenizer, AutoModelForMaskedLM
import torch

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# マスク付きテキスト
text = f'自然言語処理を習得するには、まずは{tokenizer.mask_token}から学習することである。'

# テキストをテンソルに変換
input_ids = tokenizer.encode(text, return_tensors='pt')

# マスクのインデックスを取得
masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

# 推論
result = model(input_ids)
pred_ids = result[0][:, masked_index].topk(5).indices.tolist()[0]
for pred_id in pred_ids:
    output_ids = input_ids.tolist()[0]
    output_ids[masked_index] = pred_id
    print(tokenizer.decode(output_ids))