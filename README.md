# AI Destekli Gelişmiş Tıbbi Analiz Sistemi

Bu proje, tıbbi raporlar (PDF, TXT) ve tıbbi görüntüler (JPG, PNG) için yapay zeka destekli detaylı analizler sunan bir web uygulamasıdır. Kullanıcılar, tıbbi dokümanlarını yükleyerek ön değerlendirme raporları alabilir ve geçmiş analizlerine erişebilirler. Sistem, OpenAI GPT-4 ve gelişmiş görüntü altyapısı ile çalışır.

## Özellikler

- **Çoklu Dosya Desteği:** PDF, TXT, PNG, JPG, JPEG formatlarında tıbbi rapor ve görüntü analizi
- **Gelişmiş AI Analizi:** OpenAI GPT-4 ile detaylı, markdown formatında ve Türkçe analiz raporu
- **Görüntüden Açıklama Üretimi:** Röntgen/MR/CT gibi tıbbi görüntüler için otomatik açıklama
- **Kullanıcı Hafızası:** Her kullanıcıya özel analiz geçmişi ve benzer doküman tespiti
- **Gradio Arayüzü:** Modern, kullanıcı dostu ve kolay erişilebilir web arayüzü
- **Yasal Uyarı:** Analizler yalnızca ön değerlendirme amaçlıdır, tıbbi teşhis yerine geçmez

## Kurulum

### 1. Depoyu Klonlayın
```bash
git clone <https://github.com/burakcnaksy0/AI-Destekli-Geli-mi-T-bbi-Analiz-Sistemi>
cd agent
```

### 2. Ortamı Hazırlayın
Python 3.8+ önerilir. Gerekli paketleri yüklemek için:
```bash
pip install -r requirements.txt
```

### 3. OpenAI API Anahtarınızı Ayarlayın
Proje kök dizinine `.env` dosyası oluşturun ve aşağıdaki satırı ekleyin:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Uygulamayı Başlatın
```bash
python app.py
```
Uygulama otomatik olarak tarayıcınızda açılacaktır. Eğer açılmazsa, [http://127.0.0.1:7860](http://127.0.0.1:7860) adresini ziyaret edin.

## Kullanım

1. **Doküman Analizi:**
   - "📊 Doküman Analizi" sekmesinden dosyanızı yükleyin ve "Analiz Et" butonuna tıklayın.
   - Sonuçlar markdown formatında ve detaylı şekilde ekranda gösterilir.
2. **Geçmiş Analizler:**
   - "📚 Geçmiş Analizler" sekmesinden önceki analizlerinize ulaşabilirsiniz.

## Desteklenen Dosya Türleri
- PDF: Laboratuvar sonuçları, tıbbi raporlar
- TXT: Metin formatındaki raporlar
- PNG/JPG/JPEG: Röntgen, MR, CT gibi tıbbi görüntüler

## Kullanılan Teknolojiler
- **Python**
- **Gradio**: Web arayüzü
- **OpenAI GPT-4**: Metin analizi
- **transformers (BLIP)**: Görüntüden açıklama üretimi
- **Pillow**: Görüntü işleme
- **pypdf**: PDF okuma
- **dotenv**: Ortam değişkenleri yönetimi

## Dosya Yapısı
```
app.py                # Ana uygulama dosyası
requirements.txt      # Gerekli Python paketleri
medical_memory.json   # Kullanıcı analiz geçmişi (otomatik oluşur)
images/               # Örnek tıbbi görüntüler
```

## Güvenlik ve Yasal Uyarı
- Bu sistem yalnızca ön değerlendirme amaçlıdır, tıbbi teşhis yerine geçmez.
- Sağlık kararlarınızı mutlaka uzman bir doktora danışarak alın.
- Yüklediğiniz dosyalar analiz için geçici olarak işlenir, gizliliğiniz korunur.

## Katkı ve Geliştirme
Pull request ve issue açarak katkıda bulunabilirsiniz.
