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

Merhaba, bu proje benim ödev-1 için hazırladığım tarım sorunları analizi çalışmasıdır. Çiftçilerin tarım sorunlarını doğal dil işleme (NLP) yöntemleriyle inceledim. Ziraatciyiz.biz ve Tarimziraat.com forumlarından toplanmış 200 satırlık bir veri seti kullandım. Bu projede veri setini temizledim, kelime frekanslarını analiz ettim, TF-IDF vektörleri oluşturdum ve Word2Vec modelleri eğittim. Ödevi yaparken hem teknik hem de tarım açısından çok şey öğrendim. Aşağıda projenin detaylarını, nasıl çalıştığını ve dosyaları bulabilirsiniz.
Projenin Amacı
Bu ödevde, çiftçilerin tarım sorunlarını metin tabanlı bir veri setiyle analiz etmeyi amaçladım. Veri seti, Türkiye'deki çiftçilerin gerçek sorunlarını içeriyor (örneğin, yaprak hastalıkları, verim kaybı). Amacım:
- Sorunların dil özelliklerini anlamak (örneğin, hangi kelimeler sık kullanılıyor).
- Zipf yasasına uygunluğunu test etmek.
- TF-IDF ile önemli kelimeleri bulmak.
- Word2Vec ile sorunlar arasındaki anlam ilişkilerini keşfetmek (örneğin, "sorun" kelimesi "hastalık" ile benzer mi).

Bu analiz, tarım sorunlarını otomatik sınıflandırmak veya çözüm önermek için temel oluşturabilir.
Veri Seti
Veri seti, Ziraatciyiz.biz ve Tarimziraat.com forumlarından derlenmiş. Özellikleri:
- Dosya: tarim_problemleri_veriseti.csv
- Satır Sayısı: 200
- Boyut: ~ 100 kb kadar
- Sütunlar:- sorun_id: Benzersiz ID (1'den 200'e).
- sorun_metin: Çiftçinin sorunu (örneğin, "Şeftali tarlamızda yapraklarda kahverengileşme oldu.").
- urun_turu: Ürün (Şeftali, Domates, Buğday, vb.).
- sorun_kategorisi: Sorun türü (Hastalık, Böcek, Verim Kaybı, vb.).
- ornek_cozum: Çözüm önerisi (örneğin, "Toprak analizi yaptırın.").


Örnek Satır:
1, Bu yaz şeftali tarlamızda yapraklarda kahverengileşme ve büzüşme oldu. Sulama artırdık ama düzelme yok., Şeftali, Hastalık, Organik mantar ilacı kullanın, toprak analizi yaptırın.


Kullanılan Kütüphaneler
Projeyi yazarken şu Python kütüphanelerini kullandım:
- pandas: Veri setini yüklemek ve düzenlemek.
- nltk: Metin ön işleme (tokenization, stopwords kaldırma).
- scikit-learn: TF-IDF vektörleştirme.
- gensim: Word2Vec modelleri.
- matplotlib: Zipf grafikleri çizimi.
- numpy: Matris işlemleri.

Kurulum ve Çalıştırma
Kodu çalıştırmak için şu adımları izleyin:
- Kütüphaneleri yükleyin:pip install pandas nltk scikit-learn gensim matplotlib numpy

- NLTK kaynaklarını indirin:import nltk  
nltk.download('punkt')  
nltk.download('stopwords')  
 

- Kodu çalıştırın:python tarim_sorunlari_analizi.py  


- tarim_problemleri_veriseti.csv aynı klasörde olmalı.
- Çıktılar (CSV'ler, modeller, grafikler) otomatik oluşacak.

Proje Adımları
Ödevi yaparken şu adımları takip ettim:
- Veri Yükleme: Veri setini pandas ile yükledim.
- Ön İşleme:- Metinleri küçük harfe çevirdim.
- Noktalama işaretlerini kaldırdım.
- Türkçe stopwords'ları (örneğin, "ve", "ile") temizledim.
- Basit lemmatization ve stemming uyguladım.

- Zipf Analizi: Ham ve temizlenmiş veriler için kelime frekanslarını analiz ettim, log-log grafikler çizdim.
- TF-IDF: Lemmatized ve stemmed metinler için TF-IDF matrisleri oluşturdum.
- Word2Vec: 8 parametre setiyle 16 model eğittim.
- Sonuçlar: Her adımın çıktılarını kaydettim, rapor yazdım.

Dosyalar
Repoda şu dosyalar var:
- tarim_problemleri_veriseti.csv: 200 satırlık veri seti.
- tarim_sorunlari_analizi.py: Analizleri yapan kod.
- lemmatized_sentences.csv: Lemmatized temizlenmiş metinler.
- stemmed_sentences.csv: Stemmed temizlenmiş metinler.
- tfidf_lemmatized.csv: Lemmatized TF-IDF matrisi.
- tfidf_stemmed.csv: Stemmed TF-IDF matrisi.
- lemmatized_model_*.model: 8 lemmatized Word2Vec modeli.
- stemmed_model_*.model: 8 stemmed Word2Vec modeli.
- zipf_raw.png: Ham veri Zipf grafiği.
- zipf_cleaned.png: Temizlenmiş veri Zipf grafiği.

Öğrendiklerim
Bu ödev sayesinde:
- NLP'nin tarım gibi alanlarda nasıl kullanılabileceğini gördüm.
- Türkçe metin işleme (lemmatization, stemming) zor ama faydalıymış.
- TF-IDF ve Word2Vec'in nasıl çalıştığını daha iyi anladım.
- Zipf yasasının doğal dilde neden önemli olduğunu öğrendim.

Notlar
- Word2Vec modelleri biraz büyük, hepsini yüklemek yerine birkaçını seçebilirsiniz veya models.zip yapabilirsiniz.
- Detaylı açıklamalar ve sonuçlar PDF raporda.
- Sorularınız olursa bana GitHub üzerinden yazabilirsiniz!

Teşekkürler, iyi incelemeler! 

  
Tarım Sorunları Analizi ve Benzerlik Hesaplama
Bu proje, tarım forumlarındaki sorun metinlerini kümeler ve benzerlik analizi yapar. Yapay Zeka Dersi Ödev-1 ve Ödev-2 gerekliliklerini karşılar.
Gereksinimler
•	Python 3.x
•	Kütüphaneler: pandas, nltk, scikit-learn, gensim, matplotlib
•	Kurulum: pip install pandas nltk scikit-learn gensim matplotlib
Dosya Yapısı
•	tarim_problemleri_veriseti.csv: Ham veri seti
•	tfidf_lemmatized.csv, tfidf_stemmed.csv: TF-IDF matrisleri
•	word2vec_*.model: Word2Vec modelleri
•	tarim_tam_proje.ipynb: Jupyter Notebook betiği
•	zipf_plot.png: Zipf grafiği
•	elbow_plot.png: Elbow grafiği
•	jaccard_matrix.csv: Jaccard matrisi
•	semantic_evaluation.csv: Anlamsal değerlendirme tablosu
•	clustered_problems_with_solutions.csv: Kümelenmiş sorunlar ve çözümler
Çalıştırma Talimatları
1.	Veri seti ve model dosyalarını aynı dizine yerleştirin.
2.	Jupyter Notebook’ta betiği çalıştırın:
jupyter notebook tarim_tam_proje.ipynb
3.	Çıktılar:
o	Hücrelerde: Veri seti, TF-IDF/Word2Vec sonuçları, kümelenmiş metinler, anlamsal değerlendirme, Jaccard matrisi, grafikler.
o	Dosyalar: tfidf_*.csv, jaccard_matrix.csv, semantic_evaluation.csv, clustered_problems_with_solutions.csv, zipf_plot.png, elbow_plot.png.
Notlar
•	Word2Vec modelleri yoksa betik otomatik eğitir (yavaş olabilir).
•	Anlamsal skorlar örnek olarak verilmiştir; manuel analiz yapmalısınız.






