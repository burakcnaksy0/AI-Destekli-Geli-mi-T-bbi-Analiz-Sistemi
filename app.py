import gradio as gr
import os
import json
import hashlib
from datetime import datetime
from pypdf import PdfReader
from openai import OpenAI
from transformers import pipeline
from PIL import Image
import torch
from typing import Dict, List, Any

from dotenv import load_dotenv

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API anahtarı .env dosyasında bulunamadı veya ayarlanmadı.")

client = OpenAI(api_key=api_key)
image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


# Hafıza sistemi için basit bir sınıf
class MemorySystem:
    def __init__(self, memory_file="medical_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()

    def load_memory(self) -> Dict[str, Any]:
        """Hafıza dosyasını yükler"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Hafıza yüklenirken hata: {e}")
        return {"sessions": {}, "user_history": []}

    def save_memory(self):
        """Hafızayı dosyaya kaydeder"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Hafıza kaydedilirken hata: {e}")

    def add_analysis(self, user_id: str, document_type: str, analysis_result: str, document_hash: str):
        """Yeni analiz sonucunu hafızaya ekler"""
        if user_id not in self.memory["sessions"]:
            self.memory["sessions"][user_id] = []

        analysis_record = {
            "timestamp": datetime.now().isoformat(),
            "document_type": document_type,
            "document_hash": document_hash,
            "analysis": analysis_result,
            "id": len(self.memory["sessions"][user_id]) + 1
        }

        self.memory["sessions"][user_id].append(analysis_record)
        self.save_memory()

    def get_user_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Kullanıcının geçmiş analizlerini getirir"""
        if user_id in self.memory["sessions"]:
            return self.memory["sessions"][user_id][-limit:]
        return []

    def find_similar_documents(self, user_id: str, document_hash: str) -> List[Dict]:
        """Benzer dokümanları bulur"""
        similar = []
        if user_id in self.memory["sessions"]:
            for record in self.memory["sessions"][user_id]:
                if record["document_hash"] == document_hash:
                    similar.append(record)
        return similar


# Global hafıza sistemi
memory_system = MemorySystem()


def create_document_hash(content: str) -> str:
    """Doküman içeriği için hash oluşturur"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def get_enhanced_ai_analysis(text_content: str, user_id: str, document_type: str) -> str:
    """Gelişmiş AI analizi - hafıza sistemi ile entegre"""

    # Kullanıcının geçmiş analizlerini kontrol et
    user_history = memory_system.get_user_history(user_id, 3)
    history_context = ""

    if user_history:
        history_context = "\n\n**KULLANICI GEÇMİŞİ (Son 3 analiz):**\n"
        for i, record in enumerate(user_history, 1):
            history_context += f"{i}. {record['timestamp'][:10]} - {record['document_type']}\n"

    try:
        system_prompt = (
            "Sen tıbbi doküman analizi konusunda uzman bir yapay zeka asistanısın. "
            "Görürürün kesinlikle tıbbi teşhis koymak değil, sadece ön değerlendirme yapmaktır. "
            "Analiz sonuçlarını Markdown formatında, detaylı ve anlaşılır şekilde sunacaksın. "
            "Hastalıklı durumlar tespit ettiğinde MUTLAKA doktora gidilmesi gerektiğini vurgula. "
            "Sağlıklı sonuçlarda da tedbiri elden bırakmaması gerektiğini belirt."
        )

        user_prompt = f"""Aşağıdaki tıbbi dokümanı çok detaylı bir şekilde analiz et.
        Markdown formatında, kapsamlı ve eylem odaklı bir rapor oluştur. **Tüm çıktı Türkçe olmalıdır.**

        {history_context}

        Rapor yapısı aşağıdaki başlıkları içermeli:

        ## 🏥 Genel Sağlık Durumu Değerlendirmesi
        Aşağıdaki kategorilerden birini seç ve gerekçesini açıkla:
        - **🟢 Normal / Belirgin Sorun Yok**: Bulgular normal sınırlar içinde
        - **🟡 Doktor Değerlendirmesi Gerekiyor**: Norm değerlerinden sapma var, uzman görüşü şart
        - **🔴 ACİL DOKTOR MÜDAHALESI GEREKİYOR**: Kritik bulgular mevcut
        - **⚪ Belirsiz / Yetersiz Veri**: Değerlendirme için yeterli bilgi yok

        ## 📋 Belge Türü ve Amacı
        - Doküman türü ve ne amaçla yapıldığı
        - Test/inceleme tarihi ve geçerliliği

        ## 🔍 Detaylı Bulgular Analizi
        - Ana bulgular ve sonuçlar
        - Önemli parametreler
        - Dikkat çeken noktalar

        ## ⚠️ Referans Dışı Değerler
        Her anormal değer için:
        - **Parametre Adı**: Değer (Normal Aralık: X-Y)
        - **Anlam**: Bu değerin ne ifade ettiği
        - **Önem Derecesi**: Düşük/Orta/Yüksek risk

        ## 📚 Tıbbi Terimler Sözlüğü
        Rapordaki 5-7 önemli tıbbi terim ve açıklamaları

        ## 🎯 Genel Değerlendirme ve Öneriler

        ### Durum Değerlendirmesi:
        **Sağlıklı bulgular varsa:**
        - Mevcut sağlık durumunun korunması için öneriler
        - Düzenli kontrol önemleri
        - Yaşam tarzı önerileri

        **Sorunlu bulgular varsa:**
        - ⚠️ **MUTLAKA DOKTORA GİDİN!** 
        - Hangi uzmanlık dalına başvurulmalı
        - Aciliyet derecesi
        - Doktora sorulması gereken sorular

        ### Takip Önerileri:
        - Ne sıklıkla kontrol edilmeli
        - Hangi testler tekrarlanmalı
        - Yaşam tarzı değişiklikleri

        ## 📞 Acil Durum Kriterleri
        Eğer aşağıdaki durumlardan biri varsa derhal 112'yi arayın:
        - [Kritik değerler ve durumlar]

        ---

        ### ⚖️ Yasal Uyarı
        **Bu analiz yapay zeka tarafından üretilmiş ön değerlendirmedir ve kesinlikle tıbbi teşhis yerine GEÇMEZ. Tıbbi kararlar almak için kullanılamaz. Sağlığınızla ilgili tüm kararları mutlaka uzman bir doktora danışarak alın.**

        Doküman İçeriği:
        ---
        {text_content}
        ---
        """

        response = client.chat.completions.create(
            model="gpt-4",  # Daha gelişmiş model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Daha tutarlı sonuçlar için
            max_tokens=4000,  # Daha detaylı analiz için
        )

        analysis_result = response.choices[0].message.content

        # Hafızaya kaydet
        document_hash = create_document_hash(text_content)
        memory_system.add_analysis(user_id, document_type, analysis_result, document_hash)

        return analysis_result

    except Exception as e:
        return f"## ❌ Analiz Hatası\n\nOpenAI ile analiz sırasında bir hata oluştu: {str(e)}\n\nLütfen tekrar deneyin veya dosyanızı kontrol edin."


def analyze_document_enhanced(file, user_session_id):
    """Gelişmiş doküman analizi fonksiyonu"""
    if file is None:
        return "## ⚠️ Dosya Yükleme Hatası\n\nLütfen analiz için bir dosya yükleyin."

    # Eğer session ID yoksa yeni bir tane oluştur
    if not user_session_id:
        user_session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        file_path = file.name
        filename = os.path.basename(file_path)
        content_for_ai = ""
        document_type = ""

        if filename.lower().endswith('.pdf'):
            document_type = "PDF Raporu"
            reader = PdfReader(file_path)
            text_list = [p.extract_text() for p in reader.pages if p.extract_text()]
            if not text_list:
                return f"## ❌ PDF Okuma Hatası\n\n'{filename}' dosyasından metin çıkarılamadı. Dosya taranmış resim olabilir."
            content_for_ai = "\n".join(text_list)

        elif filename.lower().endswith('.txt'):
            document_type = "Metin Raporu"
            with open(file_path, 'r', encoding='utf-8') as f:
                content_for_ai = f.read()

        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            document_type = "Tıbbi Görüntü"
            image = Image.open(file_path).convert("RGB")
            caption = image_captioner(image)[0]['generated_text']
            content_for_ai = f"Bu bir tıbbi görüntü analizidir.\n\nGörüntü açıklaması: {caption}\n\nDosya adı: {filename}"

        else:
            return f"## ❌ Desteklenmeyen Dosya Türü\n\n'{filename}' dosya türü desteklenmiyor.\n\n**Desteklenen formatlar:** PDF, TXT, PNG, JPG, JPEG"

        if content_for_ai:
            # Benzer dokümanları kontrol et
            doc_hash = create_document_hash(content_for_ai)
            similar_docs = memory_system.find_similar_documents(user_session_id, doc_hash)

            result = get_enhanced_ai_analysis(content_for_ai, user_session_id, document_type)

            if similar_docs:
                result += f"\n\n---\n## 🔄 Geçmiş Analiz Bilgisi\nBu doküman daha önce {len(similar_docs)} kez analiz edildi."

            return result, user_session_id
        else:
            return "## ❌ İçerik Hatası\n\nDosyadan analiz edilecek içerik çıkarılamadı.", user_session_id

    except Exception as e:
        return f"## ❌ İşlem Hatası\n\nDosya işlenirken hata oluştu: {str(e)}", user_session_id


def get_user_history_display(user_session_id):
    """Kullanıcının geçmişini görüntüler"""
    if not user_session_id:
        return "## 📋 Geçmiş Analiz Yok\n\nHenüz hiç analiz yapılmamış."

    history = memory_system.get_user_history(user_session_id, 10)
    if not history:
        return "## 📋 Geçmiş Analiz Yok\n\nBu oturum için henüz analiz geçmişi bulunmuyor."

    history_text = "## 📋 Geçmiş Analizleriniz\n\n"
    for i, record in enumerate(reversed(history), 1):
        date = record['timestamp'][:10]
        time = record['timestamp'][11:16]
        history_text += f"### {i}. Analiz - {date} {time}\n"
        history_text += f"**Doküman Türü:** {record['document_type']}\n\n"
        history_text += "---\n\n"

    return history_text


# --- Gradio Arayüzü ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Destekli Tıbbi Analiz Sistemi") as iface:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🏥 AI Destekli Gelişmiş Tıbbi Analiz Sistemi</h1>
        <p style="font-size: 16px; color: #666;">
            Tıbbi raporlarınızı ve görüntülerinizi yükleyerek yapay zeka destekli detaylı analiz alın
        </p>
    </div>
    """)

    # Session state için invisible textbox
    session_state = gr.Textbox(visible=False, value="")

    with gr.Tab("📊 Doküman Analizi"):
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="🔗 Tıbbi Doküman Yükle",
                    file_types=[".pdf", ".txt", ".png", ".jpg", ".jpeg"],
                    file_count="single"
                )
                analyze_btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")

                gr.HTML("""
                <div style="margin-top: 15px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                <strong>💡 Desteklenen Formatlar:</strong><br>
                • PDF: Laboratuvar sonuçları, raporlar<br>
                • TXT: Metin formatındaki raporlar<br>
                • PNG/JPG: Röntgen, MR, CT görüntüleri
                </div>
                """)

            with gr.Column(scale=2):
                analysis_output = gr.Markdown(
                    label="📋 AI Analiz Sonucu",
                    value="Analiz sonucu burada görünecektir...",
                    height=600
                )

    with gr.Tab("📚 Geçmiş Analizler"):
        history_btn = gr.Button("📋 Geçmiş Analizleri Göster", variant="secondary")
        history_output = gr.Markdown(label="Geçmiş Analizleriniz", height=500)

    # Event handlers
    analyze_btn.click(
        fn=analyze_document_enhanced,
        inputs=[file_input, session_state],
        outputs=[analysis_output, session_state]
    )

    history_btn.click(
        fn=get_user_history_display,
        inputs=[session_state],
        outputs=[history_output]
    )

    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; padding: 15px; background-color: #fff3cd; border-radius: 5px;">
        <strong>⚠️ Önemli Uyarı:</strong> Bu sistem sadece ön değerlendirme amaçlıdır. 
        Kesinlikle tıbbi teşhis yerine geçmez. Sağlık kararlarınızı mutlaka uzman doktora danışarak alın.
    </div>
    """)

if __name__ == "__main__":
    # Farklı launch seçenekleri
    try:
        # Önce localhost ile dene
        iface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False,
            inbrowser=True  # Otomatik olarak tarayıcıda aç
        )
    except Exception as e:
        print(f"Localhost ile başlatma hatası: {e}")
        print("Alternatif başlatma deneniyor...")
        # Eğer hata alırsa default ayarlarla dene
        iface.launch(
            share=False,
            debug=False,
            inbrowser=True
        )