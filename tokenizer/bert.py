from transformers import BertJapaneseTokenizer

# 日本語BERTのTokenizerを読み込む
tokenizer = BertJapaneseTokenizer.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking', force_download=True
)
# サンプルデータ
raw_inputs = [
    "私は毎週水曜日にカフェで勉強します。",
    "その後、ジムに寄ってから帰ります。",
]

# サンプルデータを入力してTokenizerによる変換を行う
tokenized_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')

print(tokenized_inputs)

converted_tokenized_inputs = [*map(lambda x: tokenizer.convert_ids_to_tokens(x), tokenized_inputs.input_ids)]
for inputs in converted_tokenized_inputs:
    print(''.join(inputs))

for input_ids in tokenized_inputs.input_ids:
    print(tokenizer.decode(input_ids).replace(' ', '')) 