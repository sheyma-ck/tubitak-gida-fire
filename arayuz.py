import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sayfa ayarları
st.set_page_config(page_title="Gıda Fire Tahmin Sistemi", page_icon="", layout="wide")

st.title("Gıda Fire Tahmin Sistemi ")
st.markdown("Lojistik parametrelerini girerek bozulma riskini yapay zeka ile hesaplayın ve israfı kaynağında çözün.")

# --- PROTOTİP MODEL EĞİTİMİ (Arka Planda) ---
ornek_X = [
    [4, 50, 2],    # Soğuk, normal nem, kısa süre -> Güvenli
    [25, 85, 7],   # Oda sıcaklığı, yüksek nem -> Bozulur
    [8, 60, 3],    # Serin, normal nem -> Güvenli
    [30, 90, 10],  # Sıcak, çok yüksek nem -> Bozulur
    [90, 50, 2],   # AŞIRI SICAK -> Bozulur
    [40, 40, 5],   # Çok sıcak, düşük nem -> Bozulur
    [2, 95, 20]    # Soğuk ama çok uzun süre ve çok nem -> Bozulur 
]
ornek_y = [0, 1, 0, 1, 1, 1, 1] 
ornek_model = RandomForestClassifier(random_state=42).fit(ornek_X, ornek_y)

# Arayüzü iki sekmeye ayıralım: Tekli Analiz ve Toplu Veri Yükleme
sekme1, sekme2 = st.tabs(["Tekil Kargo Analizi", "Toplu Veri Yükleme (Excel/CSV)"])

with sekme1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lojistik ve Finansal Parametreler")
        kategori = st.selectbox("Ürün Kategorisi Seçin", ["Taze Sebze", "Süt Ürünleri", "Kırmızı Et", "Egzotik Meyve"])
        
        kargo_agirlik = st.number_input("Kargo Ağırlığı (kg)", value=1000)
        birim_fiyat = st.number_input("Ürünün Kilo Fiyatı (TL)", value=25.0)
        
        st.markdown("---")
        sicaklik = st.number_input("Taşıma Sıcaklığı (°C)", value=4)
        nem = st.number_input("Nem Oranı (%)", value=50)
        sure = st.number_input("Taşıma Süresi (Gün)", value=2)
        
        analiz_butonu = st.button("Risk Analizi Yap")

    with col2:
        st.subheader("Yapay Zeka Karar Raporu")
        if analiz_butonu:
            tahmin = ornek_model.predict([[sicaklik, nem, sure]])
            
            finansal_deger = kargo_agirlik * birim_fiyat
            karbon_deger = kargo_agirlik * 0.15 
            
            if tahmin[0] == 1:
                st.error("🚨 YÜKSEK RİSK: Ürün yolda bozulabilir!")
                
                if sicaklik > 8:
                    sicaklik_mesaji = f"{sicaklik}°C çok sıcak! Soğutucu derecesini acilen düşürün."
                elif sicaklik < 0:
                    sicaklik_mesaji = f"{sicaklik}°C {kategori} için çok soğuk! Ürün donarak fire verebilir, sıcaklığı ideal seviyeye yükseltin."
                else:
                    sicaklik_mesaji = "Soğutucu sistemlerinizi ve nem oranını kontrol edin."

                if sure >= 5:
                    sure_mesaji = f"{sure} günlük taşıma süresi çok uzun. Lütfen rotayı daha yakın bir pazara kaydırın."
                else:
                    sure_mesaji = "Lojistik sürecini hızlandıracak alternatif rotalar planlayın."

                st.warning(f"**ÖNERİ:** {sure_mesaji} {sicaklik_mesaji}")
                st.info(f"💡 **Eylem Planı:** Yukarıdaki öneriyi uygulayıp rotayı değiştirirseniz **{finansal_deger:,.2f} TL** değerindeki ürünün israf olmasını engelleyeceksiniz!")
                
                st.markdown("**Modelin Karar Gerekçesi (Risk Faktörleri):**")
                onem_dereceleri = ornek_model.feature_importances_
                grafik_verisi = pd.DataFrame({
                    "Parametre": ["Sıcaklık", "Nem", "Süre"],
                    "Etki Oranı": onem_dereceleri
                }).set_index("Parametre")
                st.bar_chart(grafik_verisi)
                
            else:
                st.success("✅ GÜVENLİ: Ürün sağlam ulaşacaktır.")
                
                st.info("🌱 **Yeşil Ekonomi Çıktısı:**")
                st.metric(label="Korunan Finansal Değer", value=f"{finansal_deger:,.2f} TL", delta="Tasarruf Edildi")
                st.metric(label="Önlenen Karbon Salınımı", value=f"{karbon_deger:,.0f} kg", delta="- İklim Dostu", delta_color="inverse")
                
                # İŞTE EKLENEN ŞIK DİPNOTUMUZ
                st.caption("*Dipnot: Prototip arayüzündeki karbon salınımı hesaplamalarında kullanılan '1 kg gıda israfı = 0.15 kg karbon salınımı' oranı, modelin yeşil ekonomi çıktılarını görselleştirmek amacıyla kullanılan temsilî bir çarpandır.*")

with sekme2:
    st.subheader("Toplu Lojistik Verisi Analizi")
    st.markdown("İçinde yüzlerce kargonun verisi olan bir Excel/CSV dosyası yükleyin, yapay zeka sadece riskli olanları ayıklasın.")
    
    yuklenen_dosya = st.file_uploader("Kargo Veri Dosyanızı Yükleyin", type=["csv", "xlsx"])
    
    if yuklenen_dosya is not None:
        st.success("Dosya başarıyla yüklendi! Yapay zeka tüm kargoları analiz ediyor...")
        
        # 1. Yüklenen dosyayı Pandas ile okuma
        if yuklenen_dosya.name.endswith('.csv'):
            df_toplu = pd.read_csv(yuklenen_dosya)
        else:
            df_toplu = pd.read_excel(yuklenen_dosya)
            
        # 2. Excel'deki sütunları yapay zekanın anlayacağı sıraya koyma
        # (Oluşturduğumuz Excel'deki sütun isimleri: Sıcaklık_C, Nem_Oranı, Süre_Gün)
        tahmin_verisi = df_toplu[['Sıcaklık_C', 'Nem_Oranı', 'Süre_Gün']]
        
        # 3. Modele toplu tahmin yaptırma
        tahminler = ornek_model.predict(tahmin_verisi)
        
        # 4. Sonuçları (0 veya 1) tabloya yeni bir sütun olarak ekleme
        df_toplu['Yapay_Zeka_Kararı'] = ["🚨 Riskli" if t == 1 else "✅ Güvenli" for t in tahminler]
        
        # 5. Gerçek analizi ekranda gösterme
        st.dataframe(df_toplu)
