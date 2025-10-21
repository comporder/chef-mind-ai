import os
import gradio as gr
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# MODEL_CONFIG
MODEL_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 4096
}

# SYSTEM_PROMPT
SYSTEM_PROMPT = (
    "Sen deneyimli bir Türk aşçısısın. "
    "Kullanıcılara yemek tarifleri, pişirme önerileri ve püf noktaları hakkında yardımcı oluyorsun. "
    "Yanıtlarında doğal, açıklayıcı ve sıcak bir üslup kullan. "
    "Tarifleri anlaşılır adımlarla anlat, gerekirse püf noktası ver. "
    "Eğer kullanıcı malzemeler belirtirse, o malzemelere uygun bir yemek öner. "
    "Rastgele uydurma bilgi verme; sadece veri setinde bulunan bilgilerden yararlan."
)

# Hugging Face veriseti: https://huggingface.co/datasets/mertbozkurt/turkish-recipe
dataset = load_dataset("mertbozkurt/turkish-recipe", data_files="datav2.csv")

df = pd.DataFrame(dataset["train"])
df = df.sample(1000, random_state=42)
df.dropna(subset=["Title", "How-to-do"], inplace=True)
df["text"] = (
    "Yemek Adı: " + df["Title"].astype(str) + "\n"
    + "Kategori: " + df["Category"].astype(str) + "\n"
    + "Malzemeler: " + df["Materials"].astype(str) + "\n"
    + "Yapılışı: " + df["How-to-do"].astype(str)
)

model_embed = SentenceTransformer("intfloat/multilingual-e5-base")

# Tüm tariflerin embedding'lerini oluştur
embeddings = model_embed.encode(df["text"].tolist(), normalize_embeddings=True)

# FAISS ile index oluştur
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def retrieve_recipes(query, k=3):
    q_emb = model_embed.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    return df.iloc[I[0]]["text"].tolist()


def chat_function(message, history):
    try:
        context = "\n\n".join(retrieve_recipes(message))
        chat_context = "\n".join([f"Kullanıcı: {h[0]}\nAsistan: {h[1]}" for h in history])

        prompt = f"""{SYSTEM_PROMPT}
                    Geçmiş konuşma:
                    {chat_context}
                    Yeni kullanıcı mesajı: {message}
                    Aşağıda veri tabanında benzer bulunan tarifler:
                    {context}
                    Bu bilgiler ışığında, kullanıcıya uygun Türkçe bir yanıt oluştur.
                    """

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=MODEL_CONFIG,
            system_instruction=SYSTEM_PROMPT
        )

        response = model.generate_content(prompt)
        return response.text if response.text else "Yanıt alınamadı."
    except Exception as e:
        return f"Hata: {str(e)}"


theme = gr.themes.Soft(
    primary_hue="amber",         
    secondary_hue="red",
    neutral_hue="stone",
    text_size="lg",
    font=["Poppins", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #fff8f0, #ffe9d6)",
    block_background_fill="rgba(255, 255, 255, 0.85)",
    block_shadow="0 4px 20px rgba(0,0,0,0.1)",
    block_border_color="rgba(255, 200, 150, 0.5)",
    button_primary_background_fill="linear-gradient(90deg, #ff9d2f, #ff6126)",
    button_primary_background_fill_hover="linear-gradient(90deg, #ffb347, #ff6a2f)",
    button_primary_text_color="white",
    input_background_fill="rgba(255,255,255,0.9)",
    background_fill_secondary="rgba(255,255,255,0.6)",
)

demo = gr.ChatInterface(
    fn=chat_function,  
    title="👨‍🍳 ChefMind - Türkçe Yapay Zekalı Yemek Asistanı Emir ÖZALP 🍽️",
    description=(
        "👋 Hoş geldin! Ben senin kişisel Türk aşçınım. 👩‍🍳\n\n"
        "🥘 Sen elindeki malzemeleri yaz, ne pişirebileceğini birlikte bulalım.\n"
        "🍲 Ya da doğrudan bir yemeğin nasıl yapıldığını sor!\n\n"
        "ChefMind olarak sana , 11.000+ yemek tarifi arasından özel öneriler sunabilirim ✨"
    ),
    examples=[
        ["Evde tavuk, patates ve biber var, ne pişirebilirim?"],
        ["Sodalı köfte nasıl yapılır?"],
        ["Kıymalı makarna için önerin var mı?"],
        ["Tatlı istiyorum ama süt yok, ne yapabilirim?"],
        ["Hatay usulü kebap tarifi verir misin?"]
    ],
    theme=theme,
)

if __name__ == "__main__":
    demo.launch(share=True)