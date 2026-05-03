[ ufak bir öneri preview olarak değil de cod eolarak okursanız çok daha anlaşılır. ]
** LEO Uydu Yakınsama Analizi ve Çarpışma Riski Simülasyonu ** 	
	Bu proje, Alçak Dünya Yörüngesi'ndeki (LEO) uyduların çarpışma risklerini analiz etmek,
	yakınsama (conjunction) olaylarını tespit etmek ve bu kritik anları 3 boyutlu olarak simüle etmek için geliştirilmiş kapsamlı bir Uzay Bilimleri aracıdır.
	- Çanakkale Onsekiz Mart Üniversitesi (ÇOMÜ), Uzay Bilimleri ve Teknolojileri Bölümü bünyesinde yürütülen Bitirme Ödevi kapsamında geliştirilmiştir. -
	** Temel Özellikler **
	Sistem, modern yörünge mekaniği prensiplerini ve endüstri standardı olan algoritmaları
	kullanarak şu özellikleri sunar:
	· Canlı Veri Entegrasyonu: Space-Track API üzerinden Starlink, ISS ve OneWeb gibi güncel
	uydu kümelerinin TLE verilerini anlık olarak çeker.
	· Yörünge Propagasyonu: SGP4 ve SDP4 modellerini (Skyfield) kullanarak yüksek
	hassasiyetli konum ve hız tahmini yapar.
	· Gelişmiş Filtreleme: O(N^2) hesaplama yükünü azaltmak için Apsis (Apoje-Perije)
	Filtresi uygulayarak fiziksel olarak imkansız çarpışmaları eler.
	· Çarpışma Olasılığı (Pc): Foster & Estes (1992) 2D-Pc ve Chan (1997) izotropik modelleri
	ile risk hesaplar.
	· Teşhis Araçları: Mahalanobis Mesafesi testi ile 2D modellerin geçerliliğini denetler ve
	Olasılık Seyrelmesi (Probability Dilution) tespiti yapar.
	· Kinetik Enerji Analizi: Olası bir çarpışmanın sonucunu (Ec - J/g) ve Kessler Sendromu
	katkısını değerlendirir.
	· Görselleştirme: Dinamik 3B yörünge animasyonları, zemin izi (ground track) haritaları ve
	risk gösterge panelleri içerir.
	** Metodoloji **
	Proje, operasyonel uzay güvenliği (Space Situational Awareness - SSA) standartlarını takip eder:
	Veri Toplama: Space-Track GP (General Perturbations) veri tabanı.  
	Kaba Tarama: Apsis filtresi ve 5 dakikalık zaman adımlarıyla TCA (Time of Closest Approach) tespiti.  
	Risk Sınıflandırması: NASA STD-8719.14 standardına göre risk seviyelendirmesi (Kritik, Yüksek, Orta, Düşük).  
	En kötü Senaryo: Kovaryans belirsizliğine karşı Max-Pc analizi.
	** Kurulum * Projeyi yerel makinenizde çalıştırmak ** 
	1.Depoyu klonlayın:
	git clone https://github.com/kullaniciadi/leo-collision-analysis.git
	cd leo-collision-analysis
	2.Bağımlılıkları yükleyin:
	pip install -r requirements.txt
	3.Uygulamayı başlatın:
	streamlit run leo1_v2y.py
 	  	** Geliştirici **
	  	Tez Öğrencisi: Altay ÇAVUŞ
		Gözlemci Akademisyen: Doç. Dr. BURCU ÖZKARDEŞ
		Uzay Bilimleri ve Teknolojileri Bölümü - 4. Sınıf Öğrencisi
	( Not: Bu araç eğitim ve akademik araştırma amaçlıdır. Operasyonel görev kararları için resmi kurum (NASA, ESA, 18th SDS) verileri esas alınmalıdır. )
