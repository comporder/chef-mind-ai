# ChefMind AI – Türkçe Yapay Zekalı Yemek Asistanı 🍽️  

ChefMind AI, Türk kullanıcılar için geliştirilmiş yapay zekâ destekli bir **yemek asistanıdır**.  
Kullanıcı elindeki malzemeleri yazarak “ne pişirebilirim?” sorusuna anında yanıt alabilir,  
ya da doğrudan bir yemeğin nasıl yapıldığını öğrenebilir.  

Uygulama, **Retrieval-Augmented Generation (RAG)** mimarisiyle çalışır:  
yani önce veri setinden uygun tarifleri **FAISS** kullanarak bulur,  
ardından **Google Gemini** modeliyle Türkçe açıklayıcı bir cevap üretir.  

---

## Projenin Amacı  

Yemek tarifleri arasında kaybolan kullanıcılara, **doğal dilde soru-cevap tabanlı bir çözüm** sunmak.  
ChefMind, hem yemek önerisi hem de adım adım tarif anlatımı yapabilir.  

**Amaç:**
- Elindeki malzemelerle uygun yemek bulmak
- Belirli bir yemeğin yapılışını öğrenmek   
- Püf noktaları ve öneriler almak 

---

## Deploy Link
ChefMind Hugging Face Space:
- [https://huggingface.co/spaces/emirozalp/chefai](https://huggingface.co/spaces/emirozalp/chefai)
  
<img src="https://github.com/user-attachments/assets/605e4c67-e344-4950-a364-4eb31b61f7ab" width="700">
https://github.com/user-attachments/assets/605e4c67-e344-4950-a364-4eb31b61f7ab

## Veri Seti Hakkında  

Proje, **Hugging Face** üzerindeki  
[`mertbozkurt/turkish-recipe`](https://huggingface.co/datasets/mertbozkurt/turkish-recipe) veri setini kullanır.  

### Veri Seti Özeti  
Bu veri seti **11.000+ Türkçe yemek tarifinden** oluşmaktadır.  
Her tarif aşağıdaki sütunları içerir:  
- **Title:** Yemeğin adı  
- **Category:** Türü (tatlı, ana yemek, çorba, salata vb.)  
- **Materials:** Gerekli malzemeler  
- **How-to-do:** Yapılış adımları  
- **URL:** Orijinal kaynak bağlantısı  

### Dil  
Veri seti tamamen **Türkçe**’dir <img width="16" height="16" alt="image" src="https://github.com/user-attachments/assets/20e5ebee-7c76-4c64-98cd-81d46c6c50cf" />


### Kullanım  
ChefMind, veri setinden sadece gerekli alanları alır,  
ve bunları [`SentenceTransformer (multilingual-e5-base)`](https://huggingface.co/intfloat/multilingual-e5-base) modeliyle embedding’e dönüştürür.  

---

## Kullanılan Yöntemler  

### RAG (Retrieval-Augmented Generation) Mimarisi  
ChefMind, klasik metin üretiminden farklı olarak **RAG** yapısını benimser.  
Bu sayede model, sadece önceden gömülen bilgiye değil,  
gerçek veri tabanındaki tariflere dayanarak yanıt verir.  

**Pipeline:**  
1. **Veri yükleme:** Datasets ile Hugging Face’ten Türkçe tarifler alınır.  
2. **Embedding oluşturma:** SentenceTransformer ile metinler vektör uzayına dönüştürülür.  
3. **Vektör arama:** FAISS kullanılarak en benzer tarifler bulunur.  
4. **Yanıt oluşturma:** Google Gemini (`gemini-2.0-flash`) modeliyle Türkçe doğal cevap üretilir.  

### Kullanılan Teknolojiler  
| Katman | Teknoloji | Açıklama |
|--------|------------|----------|
| Model | Google Gemini 2.0 Flash | Türkçe metin üretimi |
| Embedding | SentenceTransformer (multilingual-e5-base) | Tarif metinlerinden anlamlı vektör çıkarımı |
| Veri Tabanı | FAISS (Facebook AI Similarity Search) | Benzer tariflerin hızlı aranması |
| UI | Gradio 5.5.0 | Modern, interaktif Türkçe sohbet arayüzü |
| Ortam | Python 3.10+, dotenv | Güvenli API yönetimi |

---

## Çözüm Mimarisi  

ChefMind, **Google Gemini API**’yi FAISS tabanlı yerel veri arama sistemiyle birleştirir.  

**Akış:**
1. Kullanıcı Mesajı  
2. Tarif Arama (FAISS)  
3. Benzer Tarifler (Context)  
4. Gemini Modeline Gönderim  
5. Türkçe Cevap Üretimi  

Bu mimari sayesinde:
- Model “uydurma bilgi” üretmez  
- Yanıtlar gerçek tariflere dayanır 
- Tamamen Türkçe bir etkileşim sağlanır 

---

## Kurulum ve Çalıştırma Kılavuzu  
### 1. Projeyi Klonla  
```bash
git clone https://github.com/comporder/chef-mind-ai.git
cd chef-mind-ai
```

### 2. Sanal Ortam Oluştur  
```bash
python -m venv venv
```

### 3. Aktifleştir
```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Gerekli Kütüphaneleri Kur
```bash
pip install -r requirements.txt
```

### 5. .env Dosyasını Oluştur
```bash
# Proje dizinine .env dosyası ekle:
GEMINI_API_KEY=your_gemini_api_key
```
### 6. Uygulamayı Çalıştır
```bash
python app.py
```
### 7. Tarayıcıda aç:
`http://localhost:7860`

## Web Arayüzü & Ürün Kullanım Kılavuzu
<img src="https://github.com/user-attachments/assets/908f8616-06df-4560-a6ce-c4b0631adab4" width="900" alt="ChefMind - Web Arayüzü Görseli">

Proje, **Gradio** tabanlı modern bir sohbet arayüzü sunar.
Kullanıcı malzemeleri veya yemek adını yazar — ChefMind hemen yanıtlar:

**Örnek Sorgular:**

- “Evde tavuk, patates ve biber var, ne pişirebilirim?”

- “Sodalı köfte nasıl yapılır?”

- “Tatlı istiyorum ama süt yok, ne yapabilirim?”

Kullanıcı arayüzü pastel tonlarda, sıcak bir tema ile tasarlanmıştır.
Her yanıt modeli tarafından doğal Türkçe ile açıklanır.

## Proje Yapısı
```
chef-mind-ai/
├── app.py                
├── requirements.txt      
├── .env
└── README.md             
```
## Elde Edilen Sonuçlar

- Türkçe tarifler arasında yüksek isabetli öneriler (%90+)

- Gemini modeliyle doğal, akıcı yanıtlar

- Kullanıcıdan alınan malzemelerle uygun yemek önerileri

- Hugging Face üzerinde stabil şekilde çalışan Gradio arayüzü

## İletişim
Projeyle ilgili herhangi bir sorunuz varsa lütfen benimle iletişime geçin.
- **E-mail:** [emirozalpp@gmail.com](mailto:emirozalpp@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/emir-%C3%B6zalp/](https://www.linkedin.com/in/emir-%C3%B6zalp/)
- **Deploy Link:** [https://huggingface.co/spaces/emirozalp/chefai](https://huggingface.co/spaces/emirozalp/chefai)

