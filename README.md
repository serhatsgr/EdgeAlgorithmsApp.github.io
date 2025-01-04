Kenar Algılama Algoritmaları Test Projesi Raporu
1. Başlık Sayfası
•	Proje Adı: Kenar Algılama Algoritmaları Test Projesi
•	Geliştirici Adı: Serhat SAĞIR
•	Başlangıç Tarihi: Kasım 2024
•	Bitiş Tarihi: Ocak 2025

2. İçindekiler
  •	Giriş 
    o	1.1. Projenin Amacı
    o	1.2. Kenar Algılamanın Önemi
    o	1.3. Görüntü İşleme Teknolojisinin Rolü
  •	Literatür Taraması
  •	Yöntem 
    o	3.1. Kullanılan Veri Seti ve Kütüphaneler
    o	3.2. Görüntü İşleme Teknikleri
    o	3.3. Algoritmaların Uygulanması
  •	Sonuçlar ve Tartışma
  •	Gelecek Çalışmalar
  •	Kaynaklar
  •	Ekler

3. Giriş

3.1. Projenin Amacı

Bu projenin amacı, kullanıcıların çeşitli kenar algılama algoritmalarını gerçek zamanlı olarak test edebileceği bir platform sunmaktır. Proje, hem görüntü işleme teknolojilerine aşina olmayan kullanıcılar için bir deneyim ortamı hem de algoritmaların karşılaştırmalı analizine olanak tanımayı hedeflemektedir. Kullanıcılar, seçtikleri algoritmaları farklı görüntüler üzerinde uygulayarak sonuçları anlık olarak görebilir.

3.2. Kenar Algılamanın Önemi

Kenar algılama, görüntü işleme alanında önemli bir rol oynar. Nesne tanıma, segmentasyon ve sahne analizi gibi görevlerde kullanılan bu yöntem, özellikle yapay zeka ve bilgisayarla görüntüleme alanında temel bir teknolojidir. Kenar algılama teknikleri, algoritmaların gelişimini etkileyen birçok uygulamada anahtar rol oynar; örneğin, otonom araçların çevrelerini algılaması veya yüz tanıma sistemlerinin doğruluğu.

3.3. Görüntü İşleme Teknolojisinin Rolü

Görüntü işleme teknolojisi, kullanıcıların kenar algılama algoritmalarını uygulayarak farklı durumlarda nasıl performans gösterdiğini görmelerini sağlar. Bu, hem akademik çalışmalar hem de endüstriyel uygulamalar için değerli bir araçtır. Kullanıcılar, farklı görüntü setleri ile algoritmaları test edebilir ve karşılaştırabilir

4. Literatür Taraması

4.1. Daha Önce Yapılmış Benzer Çalışmalar

Sobel, Canny ve diğer kenar algılama algoritmalarının görüntü işleme alanındaki klasik çalışmalara olan katkıları detaylandırılabilir. Örneğin, "Canny Edge Detector: Theory and Implementation" başlıklı makale, bu algoritmanın teorik temellerini ve pratik uygulamalarını ele alır. Canny algoritması, düşük ve yüksek eşik değerlerine dayanan iki aşamalı bir süreçle kenarları daha hassas bir şekilde algılar.

4.2. Kenar Algılama Algoritmaları

Sobel, Canny, Laplacian gibi algoritmaların teorik açıklamaları ve karşılaştırmaları yapılabilir. Örneğin:
•	Sobel Algoritması: X ve Y yönlerindeki gradyanları hesaplar, kenarların yönünü ve büyüklüğünü belirler. Aşağıdaki kod parçası, Sobel algoritmasının nasıl uygulandığını göstermektedir:
edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
•	Canny Algoritması: Gaussian bulanıklığı kullanarak gürültüyü azaltır ve ardından gradyan hesaplamaları yaparak kenarları belirler.
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

5. Yöntem

5.1. Kullanılan Veri Seti ve Kütüphaneler
Projenin çalışması için gerekli kütüphaneler:
  •	Flask: Web geliştirme için
  •	Flask-SocketIO: Gerçek zamanlı iletişim için
  •	OpenCV: Görüntü işleme işlemleri için
  •	NumPy: Nümerik hesaplamalar için
Bu kütüphaneler, projenin temel yapı taşlarını oluşturarak, kullanıcıların görüntüleri yükleyip işlemelerine olanak tanır.

5.2. Görüntü İşleme Teknikleri
Kullanılan algoritmaların her biri için işleme mantıkları detaylandırılabilir:

5.3 Algoritmalar ve Uygulama Yöntemleri
Kullanıcı tarafından seçilen algoritmalar, apply_algorithm fonksiyonu ile görüntüye uygulanmaktadır. Örneğin:
    def apply_algorithm(frame, algorithm):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if algorithm == 'sobel':
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        elif algorithm == 'canny':
            edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        # Diğer algoritmaların uygulanması
        return cv2.convertScaleAbs(edges)

6. Sonuçlar ve Tartışma
Algoritmaların performansı karşılaştırılabilir ve sonuçların kullanım senaryolarına göre değerlendirilmesi yapılabilir. Örneğin, Canny algoritmasının daha fazla kenar detayını koruduğu, ancak Sobel algoritmasının daha hızlı çalıştığı gözlemlenebilir. Kullanıcılar, farklı görüntülerle bu sonuçları test ederek en iyi algoritmayı seçebilir.

7. Gelecek Çalışmalar
  •	Daha fazla kenar algılama algoritması eklenmesi.
  •	Kullanıcı arayüzünün iyileştirilmesi.
  •	Sonuçların kaydedilip paylaşılabileceği bir sistem entegrasyonu.

8. Kaynaklar 
  •	https://dergipark.org.tr
  •	https://opencv.org

9. Ekler
•	Uygulamanın ekran görüntüleri.


9.1 Ekran Görüntüsü


![image](https://github.com/user-attachments/assets/d859f6ce-9bf3-4e31-8c22-c7ca20fc497e)


9.2 Ekran Görüntüsü


![image](https://github.com/user-attachments/assets/b0faab92-a87f-4236-8f73-398467f5a9cb)

 
9.3 Ekran Görüntüsü


![image](https://github.com/user-attachments/assets/aa5af715-5b58-4921-8291-ac31d132c352)


9.4 Ekran Görüntüsü


![image](https://github.com/user-attachments/assets/5611daf1-ee27-4017-8c88-91adf5cab97a)

 


