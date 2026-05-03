# 🚀 LEO Uydu Yakınsama Analizi ve Çarpışma Riski Simülasyonu

Bu proje, Alçak Dünya Yörüngesi'ndeki (LEO) uyduların çarpışma risklerini analiz etmek, yakınsama (conjunction) olaylarını tespit etmek ve bu kritik anları 3 boyutlu olarak simüle etmek için geliştirilmiş kapsamlı bir **Uzay Bilimleri** aracıdır.

Çanakkale Onsekiz Mart Üniversitesi (ÇOMÜ), Uzay Bilimleri ve Teknolojileri Bölümü bünyesinde yürütülen **Bitirme Ödevi** kapsamında geliştirilmiştir.

---

## 🛰️ Temel Özellikler

Sistem, modern yörünge mekaniği prensiplerini ve endüstri standardı olan algoritmaları kullanarak şu özellikleri sunar:

*   **Canlı Veri Entegrasyonu:** Space-Track API üzerinden Starlink, ISS ve OneWeb gibi güncel uydu kümelerinin TLE verilerini anlık olarak çeker.
*   **Yörünge Propagasyonu:** SGP4 ve SDP4 modellerini (Skyfield) kullanarak yüksek hassasiyetli konum ve hız tahmini yapar[cite: 1].
*   **Gelişmiş Filtreleme:** $O(N^2)$ hesaplama yükünü azaltmak için **Apsis (Apoje-Perije) Filtresi** uygulayarak fiziksel olarak imkansız çarpışmaları eler[cite: 1].
*   **Çarpışma Olasılığı ($P_c$):** Foster & Estes (1992) 2D-Pc ve Chan (1997) izotropik modelleri ile risk hesaplar[cite: 1].
*   **Teşhis Araçları:** Mahalanobis Mesafesi testi ile 2D modellerin geçerliliğini denetler ve **Olasılık Seyrelmesi (Probability Dilution)** tespiti yapar[cite: 1].
*   **Kinetik Enerji Analizi:** Olası bir çarpışmanın sonucunu ($E_c$ - J/g) ve Kessler Sendromu katkısını değerlendirir[cite: 1].
*   **Görselleştirme:** Dinamik 3B yörünge animasyonları, zemin izi (ground track) haritaları ve risk gösterge panelleri içerir[cite: 1].

---

## 📑 Metodoloji

Proje, operasyonel uzay güvenliği (Space Situational Awareness - SSA) standartlarını takip eder:

1.  **Veri Toplama:** Space-Track GP (General Perturbations) veri tabanı[cite: 1].
2.  **Kaba Tarama:** Apsis filtresi ve 5 dakikalık zaman adımlarıyla TCA (Time of Closest Approach) tespiti[cite: 1].
3.  **Risk Sınıflandırması:** NASA STD-8719.14 standardına göre risk seviyelendirmesi (Kritik, Yüksek, Orta, Düşük)[cite: 1].
4.  **En Kötü Senaryo:** Kovaryans belirsizliğine karşı Max-Pc analizi[cite: 1].

---

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için:

1.  **Depoyu klonlayın:**
    ```bash
    git clone [https://github.com/altonthecoder/leo-collision-analysis.git](https://github.com/altonthecoder/leo-collision-analysis.git)
    cd leo-collision-analysis
    ```

2.  **Bağımlılıkları yükleyin:**
    
```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı başlatın:**
    ```bash
    streamlit run leo1_v2y.py
    ```

---

## 👨‍💻 Proje Ekibi

**Geliştirici / Tez Öğrencisi:**  
Altay ÇAVUŞ  
*Çanakkale Onsekiz Mart Üniversitesi, Uzay Bilimleri ve Teknolojileri Bölümü*

**Akademik Danışman:**  
Doç. Dr. BURCU ÖZKARDEŞ  
*Gözlemci Akademisyen*

---
> **Not:** Bu araç eğitim ve akademik araştırma amaçlıdır. Operasyonel görev kararları için resmi kurum (NASA, ESA, 18th SDS) verileri esas alınmalıdır.
