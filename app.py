import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import math
from PIL import Image

# Sayfa Ayarları
st.set_page_config(page_title="Aesthetix Analyzer", layout="centered")

# --- MEDIAPIPE KURULUMU ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- YARDIMCI FONKSİYONLAR ---

def calculate_distance(p1, p2):
    """İki nokta arasındaki Öklid mesafesini hesaplar."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle(p1, p2):
    """İki nokta arasındaki açıyı (derece) hesaplar (Canthal Tilt için)."""
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle_rad = math.atan2(delta_y, delta_x)
    return math.degrees(angle_rad)

def get_percentile(value, mean, std_dev, direction="high"):
    """
    Basit bir normal dağılım simülasyonu ile yüzdelik dilim hesaplar.
    direction="high": Yüksek değer daha iyidir/nadirdir.
    direction="low": Düşük değer daha iyidir.
    direction="mid": Ortalamaya yakınlık iyidir.
    """
    import scipy.stats
    z_score = (value - mean) / std_dev
    percentile = scipy.stats.norm.cdf(z_score) * 100
    
    if direction == "high":
        return percentile
    elif direction == "low":
        return 100 - percentile
    else: # mid - ortalamadan sapma arttıkça skor düşer
        return 100 - (abs(0.5 - scipy.stats.norm.cdf(z_score)) * 200)

# --- ANA UYGULAMA ---

st.title("🧬 Yüz Estetiği ve Oran Analizi")
st.write("Fotoğrafınızı yükleyin, yapay zeka yüz hatlarınızı analiz etsin ve popülasyon verileriyle kıyaslasın.")

uploaded_file = st.file_uploader("Önden çekilmiş net bir fotoğraf yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Resmi Oku
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # MediaPipe İşlemi
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img_array.shape

        # --- KOORDİNATLARI AL (Landmark Indexleri MediaPipe standartıdır) ---
        # Bu indeksler yüzün spesifik anatomik noktalarıdır.
        
        # Gözler (Canthal Tilt)
        left_inner = (int(landmarks[362].x * w), int(landmarks[362].y * h))
        left_outer = (int(landmarks[263].x * w), int(landmarks[263].y * h))
        
        # Yüz Genişliği (Bizygomatic Width) - Elmacık kemikleri
        zygo_left = (int(landmarks[234].x * w), int(landmarks[234].y * h))
        zygo_right = (int(landmarks[454].x * w), int(landmarks[454].y * h))
        
        # Yüz Yüksekliği (Midface + Lower Face)
        nasion = (int(landmarks[10].x * w), int(landmarks[10].y * h)) # Alın ortası/saç çizgisi yakını
        menton = (int(landmarks[152].x * w), int(landmarks[152].y * h)) # Çene ucu

        # Çene Genişliği (Bigonial Width)
        gonion_left = (int(landmarks[58].x * w), int(landmarks[58].y * h))
        gonion_right = (int(landmarks[288].x * w), int(landmarks[288].y * h))

        # --- GÖRSELLEŞTİRME ---
        viz_img = img_array.copy()
        # Çizgileri çiz
        cv2.line(viz_img, left_inner, left_outer, (0, 255, 0), 2) # Göz açısı
        cv2.line(viz_img, zygo_left, zygo_right, (255, 0, 0), 2) # Yüz genişliği
        cv2.line(viz_img, nasion, menton, (0, 0, 255), 2) # Yüz yüksekliği
        cv2.line(viz_img, gonion_left, gonion_right, (255, 255, 0), 2) # Çene genişliği

        st.image(viz_img, caption="İşlenmiş Görüntü ve Ölçüm Noktaları", use_container_width=True)

        # --- HESAPLAMALAR ---
        
        # 1. Canthal Tilt (Göz Açısı)
        # Pozitif açı: Hunter eyes / Badem göz (Estetik kabul edilir)
        c_tilt = calculate_angle(left_inner, left_outer) * -1 # Y koordinatı ters işlediği için -1 ile çarpıyoruz
        
        # 2. FWHR (Facial Width to Height Ratio)
        # Genelde 1.7 - 2.0 arası maskülen kabul edilir.
        # Basitleştirilmiş hesaplama: Bizygomatic Width / (Nasion to Philtrum)
        # Burada tam yüz oranı kullanacağız: Width / Height
        face_width = calculate_distance(zygo_left, zygo_right)
        face_height = calculate_distance(nasion, menton)
        fwhr = face_width / face_height

        # 3. Jaw to Cheek Ratio (Çene / Elmacık Kemiği Oranı)
        jaw_width = calculate_distance(gonion_left, gonion_right)
        jaw_cheek_ratio = jaw_width / face_width

        # --- ANALİZ VE YÜZDELİK DİLİM TABLOSU ---
        
        st.subheader("📊 Analiz Sonuçları ve Popülasyon Kıyaslaması")
        st.info("Not: Bu veriler genel estetik literatüründeki ortalama değerlere dayalıdır ve tıbbi geçerliliği yoktur.")

        data = {
            "Ölçüm": ["Canthal Tilt (Göz Eğimi)", "Yüz Oranı (Width/Height)", "Çene/Elmacık Oranı"],
            "Senin Değerin": [f"{c_tilt:.1f}°", f"{fwhr:.2f}", f"{jaw_cheek_ratio:.2f}"],
            "İdeal/Ortalama": ["4° - 8° (Pozitif)", "1.35 - 1.40 (Golden)", "0.75 - 0.85"],
            "Popülasyon Yüzdesi": [
                f"%{int(get_percentile(c_tilt, 4, 3, 'high'))} (Daha pozitif)",
                f"%{int(get_percentile(fwhr, 1.35, 0.1, 'mid'))} (Altın orana yakınlık)",
                f"%{int(get_percentile(jaw_cheek_ratio, 0.8, 0.1, 'mid'))} (Uyumluluk)"
            ]
        }
        
        df = pd.DataFrame(data)
        st.table(df)

        # --- DETAYLI AÇIKLAMALAR ---
        st.markdown("---")
        st.subheader("📝 Ölçümler Ne Anlama Geliyor?")
        
        st.markdown("""
        **1. Canthal Tilt (Göz Eğimi):** Gözün dış köşesinin iç köşesine göre yüksekliğidir. Pozitif tilt (dış köşe yukarıda) genellikle daha çekici ve genç algılanır. Negatif tilt yorgun bir ifade verebilir.
        
        **2. FWHR (Yüz Genişlik/Yükseklik Oranı):**
        Yüzün ne kadar kompakt olduğunu gösterir. Yüksek FWHR değerleri (daha geniş yüzler) genellikle daha maskülen ve dominant bir algı yaratır.
        
        **3. Çene/Elmacık Oranı:**
        Çene hattının elmacık kemiklerine göre genişliğidir. 1'e ne kadar yakınsa yüz o kadar kare/dikdörtgen formundadır.
        """)

    else:
        st.error("Yüz tespit edilemedi. Lütfen ışığın iyi olduğu, yüzün net göründüğü bir fotoğraf yükleyin.")

else:
    st.info("Başlamak için yukarıdan bir fotoğraf yükleyin.")

