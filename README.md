# **Vyakyarth: A Multilingual Sentence Embedding Model for Indic Languages**  
[![Static Badge](https://img.shields.io/badge/Huggingface-Vyakyarth-yellow?logo=huggingface)](https://huggingface.co/krutrim-ai-labs/vyakyarth)	[![Static Badge](https://img.shields.io/badge/Github-Vyakyarth-green?logo=github)](https://github.com/ola-krutrim/Vyakyarth)	[![Static Badge](https://img.shields.io/badge/Krutrim_Cloud-Vyakyarth-orange?logo=data:image/png%2bxml;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAADpUlEQVRYCbVXTUhbQRDeRJqi2JSS1BQtgpCa0kiPehdNi6dWbfWgF0s9eGtPFSFG8VDMpSiCB28KQou0NwsS9NA/Dx4qNP1TUIqSmlKSFjQx4vabbXbJz8vLe2kz8GX3zc7MN2/2J/sszLichekN4A7gBZxpcLQ/0gijfQq8BFLAf5ELiBIEfgNEZgSxtA/5liw2eD4EfgJGSLVsyPcBQLFMiR3WIUAraCm6F4hFMQ2JB1afgFKI9Hw+IubVYhnQwvpSBnKZ2GfEvlgoiTMYeFNGcpnEK3AQV548gkYalbslLiGWdEtl2QbOpZ9FMzg4yGprazNVpvrr6+tseXlZy+cXlFeAAzk4i07eW29sbPB/kampqbyYGTzEyagC5wHKJG+v6lWgqamJdXV1wY2xhYUFtr1NBcwWnQqQYRJwUQK3gOeArjidTkakJMfHx6y+vp4tLi6KZ5/Px1ZWVkTf5M9tstcsP/SifFarlQcCAX50dKRm4/T0lPf19ann9vZ2Xl1dzZubm3lVVZVe2XPHxDS8k2Ra7fj4uCKSnUgkwnt7e+Uj393d5ZQUSSqV4sFgMJeo0DNxsx0tYtLR2x8eHorA4XCY19TUqECZCZAB1gDf398XtvTT0dGhbAvFh37Hip9LgKbYbDZWWVkpxtbW1tjBgdo1rKGhQegTiQQbHR1lbreb9fT0qDgtLS2qr9MR3AkYFMyW3pwkGo3yzs5OPjAwwFdXV4WOfra2tpSv3W5X+snJSaXXiU/chaeAHLu7u1VQrQ6VXhJgWyqT/v5+pZfjGu0OdEx3EZJTW1sbX1pa4pgGgZmZGT40NCTIMisgDy5MC3c4HEYSEItwlkjMQi7Cvb095etyufjc3ByfmJhQuiJxiVscREYdlN3w8DA/OTnhsVhM6YqQadndpAToKNZdiLmBvV4vTyaTYgo2Nze5xWLRCl5MR0exOv5NTcPY2Jiaf2zTYkSFxkX56RwgCQBUBUNSUVEh7OicoP3e2trKpqenGf1fGBTi8ufaPoGiULZZ+sbGRh6Px9WWk52RkZEsO514j3PJ6Zlure8BQ0E8Hg+fn58X2zIUCnG/38/r6uqM+L4Fx9/jFZ1cuQzFN8BIoFJsviJ20Xm6DqN4GZKIIqYbMCQOWL0GSnlLLR+6rVBMU0I75B4QAbSCGtF9h+99QO42dM0L3ZRp1Zr9OCWfrFu2FrW8lmuN5erOQuED7gLXAPl5TjHk5/kH9J8BdBc39Hn+BxqB1clokCTRAAAAAElFTkSuQmCC)](https://cloud.olakrutrim.com/console/inference-service?section=models&modelName=Krutrim&artifactName=Vyakyarth&artifactType=model)	[![Static Badge](https://img.shields.io/badge/Krutrim_AI_Labs-Vyakyarth-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iMzYiIGhlaWdodD0iMzYiIHZpZXdCb3g9IjAgMCAzNiAzNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjM2IiBoZWlnaHQ9IjM2IiByeD0iMTgiIGZpbGw9IiMxMEE1NTQiLz4KPHBhdGggZD0iTTI2LjQxNCAxMi41OTE5SDE5LjMzVjE1LjY0OTlDMjAuMDM0IDE1LjIzOTIgMjAuODQwNyAxNS4wMzM5IDIxLjc1IDE1LjAzMzlDMjIuNzkxMyAxNS4wMzM5IDIzLjY0MiAxNS4zNTY1IDI0LjMwMiAxNi4wMDE5QzI0Ljk3NjcgMTYuNjQ3MiAyNS4zMTQgMTcuNTQxOSAyNS4zMTQgMTguNjg1OUMyNS4zMTQgMTkuMzMxMiAyNS4xODkzIDIwLjA0OTkgMjQuOTQgMjAuODQxOUMyNC43MDUzIDIxLjYzMzkgMjQuMzE2NyAyMi40NDA1IDIzLjc3NCAyMy4yNjE5TDIxLjIgMjEuODMxOUMyMS41MzczIDIxLjM3NzIgMjEuODE2IDIwLjkwNzkgMjIuMDM2IDIwLjQyMzlDMjIuMjU2IDE5LjkzOTkgMjIuMzY2IDE5LjQ0MTIgMjIuMzY2IDE4LjkyNzlDMjIuMzY2IDE4LjM4NTIgMjIuMjQ4NyAxOC4wMDM5IDIyLjAxNCAxNy43ODM5QzIxLjc5NCAxNy41NjM5IDIxLjUwMDcgMTcuNDUzOSAyMS4xMzQgMTcuNDUzOUMyMC43OTY3IDE3LjQ1MzkgMjAuMTQ0IDE3Ljc2MTkgMjAuMTQ0IDE3Ljc2MTlDMjAuMTQ0IDE3Ljc2MTkgMTkuMTE0NyAxOC4xODcyIDE4Ljg4IDE4LjQyMTlWMjMuODU1OUgxNi4zODJWMjEuMDYxOUMxNS44OTggMjEuMzQwNSAxNS40MDY3IDIxLjU1MzIgMTQuOTA4IDIxLjY5OTlDMTQuNDI0IDIxLjg0NjUgMTMuODU5MyAyMS45MTk5IDEzLjIxNCAyMS45MTk5QzEyLjQwNzMgMjEuOTE5OSAxMS42NjY3IDIxLjc3MzIgMTAuOTkyIDIxLjQ3OTlDMTAuMzMyIDIxLjE3MTkgOS44MDQgMjAuNzI0NSA5LjQwOCAyMC4xMzc5QzkuMDEyIDE5LjU1MTIgOC44MTQgMTguODE3OSA4LjgxNCAxNy45Mzc5QzguODE0IDE3LjExNjUgOS4wMTIgMTYuNDEyNSA5LjQwOCAxNS44MjU5QzkuODA0IDE1LjIyNDUgMTAuMzU0IDE0Ljc2MjUgMTEuMDU4IDE0LjQzOTlDMTEuNzYyIDE0LjEwMjUgMTIuNTc2IDEzLjkzMzkgMTMuNSAxMy45MzM5QzEzLjkxMDcgMTMuOTMzOSAxNC4zMjEzIDEzLjk0ODUgMTQuNzMyIDEzLjk3NzlDMTUuMTU3MyAxNC4wMDcyIDE1LjQ4NzMgMTQuMDU4NSAxNS43MjIgMTQuMTMxOUwxNS41MDIgMTYuNTczOUMxNS4wMzI3IDE2LjQ1NjUgMTQuNTEyIDE2LjM5NzkgMTMuOTQgMTYuMzk3OUMxMy4yNTA3IDE2LjM5NzkgMTIuNzE1MyAxNi41MzcyIDEyLjMzNCAxNi44MTU5QzExLjk1MjcgMTcuMDc5OSAxMS43NjIgMTcuNDUzOSAxMS43NjIgMTcuOTM3OUMxMS43NjIgMTguNTI0NSAxMS45NDUzIDE4LjkyNzkgMTIuMzEyIDE5LjE0NzlDMTIuNjc4NyAxOS4zNjc5IDEzLjA3NDcgMTkuNDc3OSAxMy41IDE5LjQ3NzlDMTQuMTE2IDE5LjQ3NzkgMTQuNjU4NyAxOS4zMzg1IDE1LjEyOCAxOS4wNTk5QzE1LjYxMiAxOC43ODEyIDE2LjAzIDE4LjQ1ODUgMTYuMzgyIDE4LjA5MTlWMTIuNTkxOUg4VjEwLjE3MTlIMjYuNDE0VjEyLjU5MTlaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMjIuMDc0IDI4Ljk4MTlDMjEuNjkyNyAyOS4xNzI1IDIxLjIzOCAyOS4zNDg1IDIwLjcxIDI5LjUwOTlDMjAuMTY3MyAyOS42NzEyIDE5LjUyMiAyOS43NTE5IDE4Ljc3NCAyOS43NTE5QzE4LjA0MDcgMjkuNzUxOSAxNy4zODggMjkuNjEyNSAxNi44MTYgMjkuMzMzOUMxNi4yNDQgMjkuMDY5OSAxNS43OTY3IDI4LjY5NTkgMTUuNDc0IDI4LjIxMTlDMTUuMTM2NyAyNy43NDI1IDE0Ljk2OCAyNy4xOTI1IDE0Ljk2OCAyNi41NjE5QzE0Ljk2OCAyNS41MDU5IDE1LjM0MiAyNC42NjI1IDE2LjA5IDI0LjAzMTlDMTYuODIzMyAyMy40MTU5IDE3LjQyOTMgMjMuMDYzOSAxOC44MDggMjIuOTc1OUwxOS4wNzIgMjUuMjQxOUMxOC4zMjQgMjUuMjg1OSAxOC4yNjA3IDI1LjQyNTIgMTcuOTgyIDI1LjY1OTlDMTcuNzAzMyAyNS45MDkyIDE3LjU2NCAyNi4xOTUyIDE3LjU2NCAyNi41MTc5QzE3LjU2NCAyNy4xOTI1IDE4LjAxMTMgMjcuNTI5OSAxOC45MDYgMjcuNTI5OUMxOS4yNDMzIDI3LjUyOTkgMTkuNTg4IDI3LjQ3ODUgMTkuOTQgMjcuMzc1OUMyMC4yOTIgMjcuMjczMiAyMC43MTczIDI3LjA5NzIgMjEuMjE2IDI2Ljg0NzlMMjIuMDc0IDI4Ljk4MTlaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K)](https://ai-labs.olakrutrim.com/models/Vyakyarth-1-Indic-Embedding)

Vyakyarth is an advanced **sentence embedding model** designed for **Indic languages**, built upon the **STSB-XLM-R-Multilingual** architecture. Fine-tuned using **contrastive learning**, Vyakyarth excels in **cross-lingual sentence similarity, semantic search, multilingual retrieval, and text clustering**.  

[![Vyakyarth](https://img.youtube.com/vi/N1f8IlZCUi4/0.jpg)](https://www.youtube.com/watch?v=N1f8IlZCUi4)

### **🔹 Why Vyakyarth?**  
- **Supports 10+ Indic Languages** (Hindi, Tamil, Bengali, Telugu, Marathi, Kannada, Malayalam, Gujarati, Sanskrit, etc.)  
- **Optimized for Cross-Lingual Understanding** (beyond basic translation)  
- **Fast & Efficient Embeddings** for large-scale NLP applications  
- **Seamless Integration** with `sentence-transformers` and **Hugging Face Hub**  

---  

## **🛠 Installation Guide**  
To set up and run **Vyakyarth**, follow these steps:

### **1️⃣ Clone the Repository **  
If using a GitHub repository:  
```bash
git clone https://github.com/ola-krutrim/Vyakyarth.git
cd vyakyarth
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

This will install:
```
sentence-transformers>=2.2.2
scikit-learn>=1.2.2
numpy>=1.21.6
```

### **5️⃣ Run Inference**  
To test the model, run:  
```bash
python3 inference.py
```

🚀 **Vyakyarth is now ready for multilingual NLP applications!** 🎯  

---

## **📊 Performance Benchmark**  

Vyakyarth has been benchmarked on the **Flores Cross-Lingual Sentence Retrieval Task** from the **IndicXTREME Benchmark**, demonstrating superior performance in multilingual sentence similarity tasks. Compared to existing models like **MuRIL**, **IndicBERT**, and **jina-embeddings-v3**, Vyakyarth consistently achieves **higher accuracy in retrieving semantically similar sentences across Indic languages**.

| **Language**  | **MuRIL** | **IndicBERT** | **Vyakyarth** | **jina-embeddings-v3** |
|--------------|------|----------|----------|----------------|
| **Bengali**  | 77.0  | 91.0 | **98.7** | 97.4 |
| **Gujarati** | 67.0  | 92.4 | **98.7** | 97.3 |
| **Hindi**    | 84.2  | 90.5 | **99.9** | 98.8 |
| **Kannada**  | 88.4  | 89.1 | **99.2** | 96.8 |
| **Malayalam**| 82.2  | 89.2 | **98.7** | 96.3 |
| **Marathi**  | 83.9  | 92.5 | **98.8** | 97.1 |
| **Sanskrit** | 36.4  | 30.4 | **90.1** | 84.1 |
| **Tamil**    | 79.4  | 90.0 | **97.9** | 95.8 |
| **Telugu**   | 43.5  | 88.6 | **97.5** | 97.3 |


**Use Cases**

#### **1\. Natural Language Understanding**

Vyakyarth enhances virtual assistants, AI chatbots, and automated response systems by ensuring accurate intent recognition and multilingual interaction.

#### **2\. Cross-Lingual Semantic Search**

Search engines and knowledge bases benefit from Vyakyarth’s ability to retrieve contextually relevant results across multiple languages, moving beyond traditional keyword-based search.

#### **3\. Multilingual Recommendation Systems**

Vyakyarth powers content recommendations for e-commerce, OTT platforms, and news aggregators, enhancing engagement by understanding user preferences across languages.

#### **4\. AI-Powered Customer Support**

Businesses can automate multilingual customer support with high intent accuracy, reducing the need for language-specific training.

#### **5\. Content Moderation & Sentiment Analysis**

Vyakyarth ensures effective detection of toxic, misleading, or inappropriate content in multiple languages, making it essential for social media and content platforms.

# **How to Use Vyakyarth on Krutrim Cloud**

```
import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


krutrim_api_key = <krutrim_api_key>
krutrim_api_base = "https://cloud.olakrutrim.com/v1"

client = OpenAI(
   api_key=os.environ.get("KRUTRIM_API_KEY", krutrim_api_key),
   base_url=os.environ.get("KRUTRIM_BASE_URL", krutrim_api_base),
)

# Function to get embeddings
def get_embedding(sentence):
   response = client.embeddings.create(
       model="Bhasantarit-mini",
       input=sentence
   )
   return response.data[0].embedding 

# Compute cosine similarity
def cosine_sim(emb1, emb2):
   return cosine_similarity([emb1], [emb2])[0][0]


# ========= Test examples =========

# Test example 1 - Hindi
# Result:
# Similarity Score (Similar Sentences): 0.97
# Similarity Score (Dissimilar Sentences): 0.10
similar_sentence_1 = "आज मौसम बहुत सुहाना है।"  # "Today's weather is very pleasant."
similar_sentence_2 = "मौसम आज बहुत अच्छा है।"  # "The weather is very good today."

dissimilar_sentence_1 = "मैं फुटबॉल खेलना पसंद करता हूँ।"  # "I like to play football."
dissimilar_sentence_2 = "यह किताब बहुत रोचक है।"  # "This book is very interesting."

# Get embeddings
embedding_sim_1 = np.array(get_embedding(similar_sentence_1))
embedding_sim_2 = np.array(get_embedding(similar_sentence_2))
embedding_dis_1 = np.array(get_embedding(dissimilar_sentence_1))
embedding_dis_2 = np.array(get_embedding(dissimilar_sentence_2))

similarity_score_sim = cosine_sim(embedding_sim_1, embedding_sim_2)
similarity_score_dis = cosine_sim(embedding_dis_1, embedding_dis_2)
print(f"Similarity Score (Similar Sentences): {similarity_score_sim:.2f}")
print(f"Similarity Score (Dissimilar Sentences): {similarity_score_dis:.2f}")

# Classification
threshold = 0.8  # Define threshold for similarity
if similarity_score_sim > threshold:
   print("Sentence 1:" + similar_sentence_1)
   print("Sentence 2:" + similar_sentence_2)
   print("Similar Sentences: They are classified as similar ✅")
else:
   print("Sentence 1:" + dissimilar_sentence_1)
   print("Sentence 2:" + dissimilar_sentence_2)
   print("Similar Sentences: They are not classified as similar ❌")

if similarity_score_dis < 0.5:
   print("Dissimilar Sentences: They are correctly classified as different ✅")
else:
   print("Dissimilar Sentences: They are not classified correctly ❌")
```

**Licensing**

[License](LICENSE.md)



