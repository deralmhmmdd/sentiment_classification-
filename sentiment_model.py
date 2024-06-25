import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify

# Inisialisasi Flask app
app = Flask(__name__)

# Inisialisasi tokenizer dan model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Muat model yang telah dilatih
model.load_state_dict(torch.load('bert_sentiment_model.pth'))
model.eval()

# Menentukan device (CPU atau GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Fungsi untuk tokenisasi dan padding
def tokenize_and_pad(sentences, max_length=64):
    encoded_dict = tokenizer.encode_plus(
        sentences,                      # Kalimat yang akan ditokenisasi
        add_special_tokens=True,        # Tambahkan '[CLS]' dan '[SEP]'
        max_length=max_length,          # Padding & truncation length
        padding='max_length',           # Pad ke max_length
        return_attention_mask=True,     # Return attention mask
        return_tensors='pt',            # Return pytorch tensors
        truncation=True                 # Aktifkan truncation
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    return input_ids, attention_mask

# Mapping kategori label ke teks
label_mapping = {0: 'negative', 1: 'positive', 2: 'neutral'}

# Endpoint untuk klasifikasi sentimen
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan input teks dari request
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Tokenisasi dan padding input teks
    input_ids, attention_mask = tokenize_and_pad(text)

    # Lakukan prediksi
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs.logits

    # Mendapatkan prediksi label
    preds = torch.argmax(logits, dim=1).flatten()
    sentiment = label_mapping[preds.item()]

    # Mengembalikan hasil dalam format JSON
    return jsonify({"output": sentiment})

# Menjalankan Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
