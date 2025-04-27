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

