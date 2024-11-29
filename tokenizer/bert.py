from transformers import AutoTokenizer

# 日本語BERTのTokenizerを読み込む
tokenizer_jp = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# サンプルデータ
raw_inputs = [
    "私は毎週水曜日にカフェで勉強します。",
    "その後、ジムに寄ってから帰ります。",
]

# サンプルデータを入力してTokenizerによる変換を行う
tokenized_inputs = tokenizer_jp(raw_inputs, padding=True, truncation=True, return_tensors='pt')

print(tokenized_inputs)

converted_tokenized_inputs = [*map(lambda x: tokenizer_jp.convert_ids_to_tokens(x), tokenized_inputs.input_ids)]
for inputs in converted_tokenized_inputs:
    print(''.join(inputs))

for input_ids in tokenized_inputs.input_ids:
    print(tokenizer_jp.decode(input_ids).replace(' ', '')) 