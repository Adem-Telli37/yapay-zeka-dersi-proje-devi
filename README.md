# yapay-zeka-dersi-proje-devi Tarım Projesi: Ürün Yetiştiricilik Sorunlarının Benzerlik Analizi
 Tarım Projesi: Ürün Yetiştiricilik Sorunlarının Benzerlik Analizi
 Tarım Projesi - Yapay Zeka Ödevi: Aşamalar ve Açıklamalar
1. Kütüphanelerin Yüklenmesi ve Hazırlık
Pandas, NLTK, re, csv, wordcloud, matplotlib gibi Python kütüphaneleri projeye dahil edildi.

Özellikle wordcloud kütüphanesi eksikti, sonradan yüklendi (pip install wordcloud komutu ile).

NLTK içindeki Türkçe dil desteği için gerekli veri setleri indirildi (punkt).

2. Verinin Yüklenmesi
Elimizde bir tarım şikayetleri veri seti vardı.

Bu veriler .csv dosyasından bir pandas DataFrame (df) olarak yüklendi.

sorun_metin adlı sütun kullanılarak metinler alındı.

3. Türkçe Stopwords Belirlenmesi
Anlam taşımayan ("ve", "ile", "da", "bu", "şu" gibi) Türkçe kelimeler bir stopwords listesine eklendi.

Bu kelimeler daha sonra veriden ayıklanmak üzere hazırlandı.

4. Metinlerin Temizlenmesi ve İlk Analizler
Verinin ilk 500 karakteri yazdırılarak veriye hızlı bir göz atıldı.

Cümleler cümle cümle bölündü (sentence tokenization yapıldı).

İlk 10 cümle çıkarılıp örnek gösterildi.

5. Ön İşlem: Lemmatization ve Stemming
Her cümledeki kelimeler:

Lemmatize edildi (kelimenin kök anlamlı hali).

Stem edildi (kelimenin kök biçimi).

Örnek bir cümle üzerinde hem lemmatize hem stem yapılmış hali gösterildi.

Örneğin:

Ham: "Bu yaz şeftali tarlamızda yapraklarda kahverengileşme ve büzüşme."

Lemmatized: ['yaz', 'şeftali', 'tarlamızda', 'yapraklarda', 'kahverengileşme', 'büzüşme']

Stemmed: ['yaz', 'şeftali', 'tarlamız', 'yapraklar', 'kahverengileşm', 'büzüşm']

6. Temizlenmiş Verinin Karşılaştırılması
İlk 5 cümle için hem ham, hem lemmatized hem de stemmed versiyonları yazdırıldı.

Böylece temizleme işleminin veriye etkisi gözlemlendi.

7. Word Cloud (Kelime Bulutu) Oluşturulması
Temizlenen (lemmatized) metinler birleştirildi.

Bu metinler kullanılarak WordCloud (kelime bulutu) oluşturuldu.

matplotlib kullanılarak bu kelime bulutu görselleştirildi.

8. Veri Boyutu Analizi
Ham veri ve temizlenmiş veri karakter sayıları karşılaştırıldı.

Boyut küçülmesi hesaplandı.

Örneğin: yaklaşık %6.88'lik bir veri küçülmesi sağlandı.

 Genel Değerlendirme
Bu proje ile:

Tarımsal şikayet verileri temizlendi ve ön işlendi.

Türkçe doğal dil işleme adımları uygulandı (stopwords temizleme, lemmatization, stemming).

Görselleştirme (WordCloud) ile en çok kullanılan kelimeler belirlendi.

Veri boyutu optimize edilerek gereksiz kelimelerden arındırıldı.


Tarım Sorunları Analizi Ödevi
Bu proje, çiftçilerin karşılaştığı tarım sorunlarını doğal dil işleme (NLP) yöntemleriyle analiz etmeyi amaçlıyor. Çalışmada, Ziraatciyiz.biz ve Tarimziraat.com forumlarından toplanmış 200 satırlık veri seti kullanıldı. Veri temizleme, kelime frekans analizi, TF-IDF vektörleme ve Word2Vec modeli eğitme gibi adımlar izlendi.
Proje Amacı
Bu çalışmayla:
- Çiftçilerin sorunlarını metinsel veriler üzerinden analiz etmek,
- Kullanılan dilin yapısını anlamak,
- Önemli kelimeleri belirlemek (TF-IDF),
- Sorunlar arasındaki anlam ilişkilerini keşfetmek (Word2Vec) hedeflenmiştir.

Veri Seti
Veri, çiftçi forumlarından toplanarak düzenlenmiştir. Özellikler:
- Dosya Adı: tarim_problemleri_veriseti.csv
- Satır Sayısı: 200
- Sütunlar:- sorun_id: Benzersiz ID
- sorun_metin: Çiftçinin yazdığı metin
- urun_turu: Ürün adı (Domates, Buğday vb.)
- sorun_kategorisi: Sorun türü (Hastalık, Zararlı vb.)
- ornek_cozum: Önerilen çözüm


Kullanılan Kütüphaneler
Projede şu Python kütüphaneleri kullanılmıştır:
- pandas (veri işleme)
- nltk (metin ön işleme)
- scikit-learn (TF-IDF)
- gensim (Word2Vec)
- matplotlib (grafikler)
- numpy (matris işlemleri)

Kurulum & Çalıştırma
Kodun çalışması için aşağıdaki adımlar takip edilmelidir:
- Kütüphaneleri yükleyin:pip install pandas nltk scikit-learn gensim matplotlib numpy

- NLTK kaynaklarını indirin:import nltk
nltk.download('punkt')
nltk.download('stopwords')


- Analiz kodunu çalıştırın:python tarim_sorunlari_analizi.py


Proje Adımları
Çalışmada aşağıdaki adımlar takip edilmiştir:

- Veri setinin yüklenmesi
  
- Metin ön işleme (küçük harfe çevirme, noktalama kaldırma, lemmatization)
  
- Zipf Yasası analizi
  
- TF-IDF ile önemli kelimelerin belirlenmesi
 
- Word2Vec modelleri eğitilmesi
  

Dosyalar
Projede şu dosyalar yer almaktadır:
- tarim_problemleri_veriseti.csv: Veri seti
- tarim_sorunlari_analizi.py: Analiz kodları
- TF-IDF, Word2Vec çıktıları ve grafikler

Öğrenilenler
Bu proje sayesinde:
- NLP’nin tarım sorunlarını analiz etmek için nasıl kullanılabileceği görüldü,
- Türkçe metin işleme konusunda deneyim kazanıldı,
- Zipf yasasının doğal dildeki önemi keşfedildi.




