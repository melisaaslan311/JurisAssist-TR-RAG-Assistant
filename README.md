# ⚖️ Turkish Labor Law AI Assistant (İş Kanunu RAG Asistanı)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC2626?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

**Turkish Labor Law AI Assistant**, 4857 Sayılı Türk İş Kanunu metinleri üzerinde yüksek doğrulukla anlamsal arama (semantic search) yaparak kullanıcılara madde dayanaklı hukuki yanıtlar üreten **RAG (Retrieval-Augmented Generation)** tabanlı bir yapay zeka asistanıdır.

---

## 📌 Temel Özellikler

- **Kural Tabanlı Madde Ayrıştırma (Rule-Based Chunking):** Kanun metnini rastgele karakter/kelime sınırlarına göre değil, madde bütünlüğünü ve başlıklarını koruyarak böler.
- **Çok Dilli Anlamsal Vektörleştirme:** Türkçe hukuki terminolojiye uyumlu `multilingual-e5-large` embedding modeli ile yüksek doğruluklu vektör temsili.
- **Bulut Tabanlı Vektör Arama:** Düşük gecikme süreli ve ölçeklenebilir arama için **Qdrant Cloud** entegrasyonu.
- **Sıfır Halüsinasyon Prensibi:** Modele verilen bağlam dışına çıkmama ve bilgi bulunamadığında bunu doğrudan belirtme talimatı.
- **Şeffaf Kaynak Gösterimi:** Her cevabın altında kullanılan kanun maddelerini ve içeriklerini gösteren açılır panel (*expander*).
- **Akıcı Kullanıcı Deneyimi:** Streamlit ile gerçek zamanlı yanıt akışı (*streaming response*).

---

## 🏗️ Mimari Şema

```text
[ 4857 Sayılı İş Kanunu PDF ]
              │
              ▼ (PyPDFLoader & Regex Chunking)
   [ Madde Bazlı Chunk'lar ]
              │
              ▼ (multilingual-e5-large)
   [ 1024-d Vektör Embedding ]
              │
              ▼
   [ Qdrant Cloud Veritabanı ]
              │
══════════════╪══════════════════════════════════════
              │ 🔍 Kullanıcı Sorusu (Streamlit UI)
              ▼
   [ Semantik Benzerlik Araması (Top-K) ]
              │
              ▼
   [ İlgili Kanun Maddeleri (Bağlam) ] + [ Sistem Promptu ]
              │
              ▼
   [ Google Gemini 3.6 Flash ]
              │
              ▼
[ Akıcı & Madde Referanslı Hukuki Yanıt ]
