# ChefMind - Türkçe Yapay Zekalı Yemek Asistanı
# Proje Sahibi: Emir ÖZALP
# Açıklama: Bu proje, Google Gemini destekli bir Türkçe yemek asistanıdır.
# Kullanıcıdan alınan girdiye göre veri setinden benzer tarifleri FAISS kullanarak bulur
# ve Gemini modeliyle doğal, açıklayıcı şekilde yanıtlar üretir.

# GEREKLİ KÜTÜPHANELERİN İÇE AKTARILMASI
import os
import gradio as gr
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv


# ORTAM DEĞİŞKENLERİNİN YÜKLENMESİ (.env dosyasından)
# Burada .env dosyasındaki API anahtarımı okutarak Google Gemini modelini yetkilendiriyorum.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


# MODEL YAPILANDIRMASI
# Burada modelin üretim (generation) parametrelerini belirliyorum.
# temperature: cevabın yaratıcılığını kontrol eder
# top_p, top_k: örnekleme çeşitliliğini sınırlar
# max_output_tokens: maksimum cevap uzunluğunu token bazında belirler
MODEL_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 4096
}


# SİSTEM PROMPT’U (MODELİN ROL TANIMI)
# Burada modelin nasıl davranacağını tanımlıyorum.
# Kısaca: Model bir Türk aşçısı gibi konuşacak, doğru, doğal ve açıklayıcı olacak.
SYSTEM_PROMPT = (
    "Sen deneyimli bir Türk aşçısısın. "
    "Kullanıcılara yemek tarifleri, pişirme önerileri ve püf noktaları hakkında yardımcı oluyorsun. "
    "Yanıtlarında doğal, açıklayıcı ve sıcak bir üslup kullan. "
    "Tarifleri anlaşılır adımlarla anlat, gerekirse püf noktası ver. "
    "Eğer kullanıcı malzemeler belirtirse, o malzemelere uygun bir yemek öner. "
    "Rastgele uydurma bilgi verme; sadece veri setinde bulunan bilgilerden yararlan."
)


# VERİ SETİNİN YÜKLENMESİ
# Burada Hugging Face’te yer alan “mertbozkurt/turkish-recipe” veri setini yüklüyorum.
# Bu veri seti 11.000+ Türkçe yemek tarifinden oluşuyor.
dataset = load_dataset("mertbozkurt/turkish-recipe", data_files="datav2.csv")

# Dataset'i DataFrame formatına çeviriyorum.
df = pd.DataFrame(dataset["train"])

# Hugging Face Spaces üzerinde (free tier kullanıcı özelliklerinden ötürü) daha hızlı başlatmak için veri setinin belirli bir kısmını alıyorum.
df = df.sample(1000, random_state=42)

# Eksik tarifleri temizliyorum (Title ve How-to-do sütunları boş olanları siliyorum).
df.dropna(subset=["Title", "How-to-do"], inplace=True)

# Her tarif için tek bir metin alanı oluşturuyorum.
# Bu alanda yemek adı, kategori, malzemeler ve yapılış bilgisi birleştiriliyor.
df["text"] = (
    "Yemek Adı: " + df["Title"].astype(str) + "\n"
    + "Kategori: " + df["Category"].astype(str) + "\n"
    + "Malzemeler: " + df["Materials"].astype(str) + "\n"
    + "Yapılışı: " + df["How-to-do"].astype(str)
)


# EMBEDDING MODELİNİN YÜKLENMESİ
# Burada SentenceTransformer kullanarak tarif metinlerini sayısal vektörlere dönüştürüyorum.
# multilingual-e5-base modeli, çok dilli metinleri anlamlı vektörlere çeviriyor.
model_embed = SentenceTransformer("intfloat/multilingual-e5-base")

# Tüm tariflerin embedding’lerini oluşturuyorum ve normalize ediyorum.
embeddings = model_embed.encode(df["text"].tolist(), normalize_embeddings=True)


# FAISS İLE BENZERLİK ARAMA
# FAISS (Facebook AI Similarity Search) hızlı vektör araması sağlar.
# Burada cosine similarity’e denk gelen inner product (dot product) kullanıyorum.
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


# TARİF GETİRME FONKSİYONU
# Kullanıcıdan gelen soruya en benzer tarifleri embedding uzayında arıyorum.
def retrieve_recipes(query, k=3):
    q_emb = model_embed.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    return df.iloc[I[0]]["text"].tolist()


# ANA CHAT FONKSİYONU
# Bu fonksiyon, Gradio arayüzünden gelen her mesaj için çalışır.
# 1. Kullanıcının sorusuna göre benzer tarifleri bulur.
# 2. Bu tarifleri Gemini modeline bağlam olarak verir.
# 3. Modelden Türkçe, doğal bir yanıt döndürür.
def chat_function(message, history):
    try:
        # Kullanıcı mesajına göre benzer tarifleri getiriyorum.
        context = "\n\n".join(retrieve_recipes(message))

        # Sohbet geçmişini düzenliyorum (modelin bağlamı koruyabilmesi için).
        chat_context = "\n".join([f"Kullanıcı: {h[0]}\nAsistan: {h[1]}" for h in history])

        # Model prompt’unu oluşturuyorum.
        prompt = f"""{SYSTEM_PROMPT}
                    Geçmiş konuşma:
                    {chat_context}
                    Yeni kullanıcı mesajı: {message}
                    Aşağıda veri tabanında benzer bulunan tarifler:
                    {context}
                    Bu bilgiler ışığında, kullanıcıya uygun Türkçe bir yanıt oluştur.
                    """

        # Gemini modelini başlatıyorum.
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=MODEL_CONFIG,
            system_instruction=SYSTEM_PROMPT
        )

        # Cevabı üretiyorum.
        response = model.generate_content(prompt)
        return response.text if response.text else "Yanıt alınamadı."
    except Exception as e:
        return f"Hata: {str(e)}"


# GRADIO ARAYÜZ TASARIMI
# Burada Gradio’nun Soft temasını özelleştiriyorum.
# Arkaplanda yumuşak turuncu tonlar, cam efekti bloklar ve gradient butonlar kullanıldı.
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


# GRADIO CHAT INTERFACE
# ChatInterface, kullanıcıyla etkileşimi sağlayan bileşen.
# Burada örnek sorular, başlık, açıklama ve tema ayarlarını tanımlıyorum.
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


# UYGULAMA BAŞLATMA
# Eğer bu dosya doğrudan çalıştırılıyorsa (import edilmediyse),
# Gradio arayüzünü başlatıyorum. share=True, bağlantıyı dış dünyayla paylaşmamı sağlıyor.
if __name__ == "__main__":
    demo.launch(share=True)
