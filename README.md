# ⚖️ Turkish Labor Law AI Assistant

### İş Kanunu RAG Asistanı

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC2626?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Google%20Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

---

## 📌 Proje Hakkında

**Turkish Labor Law AI Assistant**, 4857 Sayılı Türk İş Kanunu metinleri üzerinde çalışan, **Retrieval-Augmented Generation (RAG)** mimarisine sahip yapay zekâ destekli bir hukuk asistanıdır.

Sistem, kullanıcının sorusunu anlamsal olarak analiz ederek Qdrant üzerinde depolanan ilgili kanun maddelerini getirir ve bu maddeleri Google Gemini modeline bağlam olarak sunar.

Böylece yanıtlar doğrudan ilgili kanun maddelerine dayandırılır ve kullanılan kaynaklar kullanıcıya açık şekilde gösterilir.

🌐 **Canlı Demo:**  
https://juris-assist-rag-assistant.streamlit.app/

---

## ✨ Temel Özellikler

### 📚 Madde Bazlı Chunking

4857 Sayılı İş Kanunu metni, rastgele karakter veya kelime uzunluklarına göre değil, kanun maddelerinin bütünlüğü korunacak şekilde parçalanır.

Her chunk ilgili madde numarası ve kanun bilgisi gibi metadata alanlarıyla birlikte saklanır.

### 🧠 Çok Dilli Semantic Embedding

Metinlerin anlamsal vektör temsillerini oluşturmak için:

**`intfloat/multilingual-e5-large`**

embedding modeli kullanılmaktadır.

Model, Türkçe dahil çok dilli metinlerde anlamsal benzerlik tabanlı arama yapılmasına olanak sağlar.

Embedding vektörleri normalize edilerek cosine similarity tabanlı arama için uygun hale getirilir.

### 🔎 Semantic Search

Kullanıcının sorusu embedding modeline aktarılır ve Qdrant üzerinde anlamsal benzerlik araması gerçekleştirilir.

En alakalı **Top-K kanun maddesi** alınarak Gemini modeline bağlam olarak gönderilir.

### ☁️ Qdrant Cloud

Kanun maddelerinin embedding vektörleri **Qdrant Cloud** üzerinde saklanır.

Qdrant, yüksek boyutlu vektörler üzerinde hızlı similarity search gerçekleştirmek için kullanılmaktadır.

### 🤖 Retrieval-Augmented Generation

Sistem klasik bir LLM chatbot yerine RAG mimarisi kullanmaktadır.

Model doğrudan soruyu cevaplamak yerine:

1. Kullanıcı sorusunu alır.
2. İlgili kanun maddelerini Qdrant'tan getirir.
3. Getirilen maddeleri bağlam olarak Gemini'ye gönderir.
4. Gemini yalnızca sağlanan bağlam üzerinden yanıt üretir.

### 🛡️ Bağlamla Sınırlandırılmış Yanıt

Prompt içerisinde modele, yalnızca getirilen kanun metinlerini kullanması ve verilen bağlamda bilgi bulunmuyorsa bunu açıkça belirtmesi söylenmektedir.

Bu yaklaşım, modelin konu dışı bilgi üretmesini azaltmayı amaçlamaktadır.

### 📖 Kaynak Gösterimi

Her yanıtın altında, kullanılan kanun maddeleri kullanıcı tarafından görüntülenebilir.

Kaynak panelinde:

- Kanun adı
- Madde numarası
- İlgili kanun metni

gösterilir.

### ⚡ Streaming Response

Gemini'den gelen yanıt Streamlit üzerinde streaming olarak gösterilir.

Bu sayede kullanıcı yanıtın tamamının oluşturulmasını beklemek yerine yanıtı oluşturulurken görebilir.

---

# 🏗️ Sistem Mimarisi

```text
                    ┌─────────────────────────┐
                    │  4857 Sayılı İş Kanunu  │
                    │          PDF            │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ PDF Text Extraction     │
                    │      PyPDFLoader        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Rule-Based Chunking     │
                    │ Madde Bazlı Ayrıştırma  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ multilingual-e5-large   │
                    │   1024-dimensional      │
                    │       Embeddings        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Qdrant Cloud       │
                    │    Vector Database      │
                    └────────────┬────────────┘
                                 │
═════════════════════════════════╪══════════════════════════════════
                                 │
                         Kullanıcı Sorusu
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Streamlit UI       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Query Embedding         │
                    │ multilingual-e5-large  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Qdrant Similarity Search│
                    │         Top-K           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Relevant Law Articles   │
                    │        Context          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Prompt Template     │
                    │ + Retrieved Context     │
                    │ + User Question         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Google Gemini       │
                    │      LLM Response       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Source-Grounded Answer  │
                    │ + Article References    │
                    └─────────────────────────┘
