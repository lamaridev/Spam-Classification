from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder

# Charger le modèle et le tokenizer
model_path = "/content/model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Assurer que le modèle est en mode évaluation
model.eval()


def predict(text):
    # Tokeniser le texte
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Obtenir les prédictions du modèle
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        predicted_class_index = torch.argmax(logits, dim=1).item()

    # Retournez directement 'spam' ou 'non-spam' selon le predicted_class_index
    return 'spam' if predicted_class_index == 1 else 'non-spam'



while True:
    user_input = input("You: ")
    predicted_category = predict(user_input)
    print(f"Predicted Category : {predicted_category}")
