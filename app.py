import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import math
from PIL import Image

# Sayfa Ayarları
st.set_page_config(page_title="Pro Aesthetix Analyzer", layout="wide")

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
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle(p1, p2, p3):
    """Üç nokta arasındaki açıyı hesaplar (p2 köşe noktasıdır)."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_tilt(p1, p2):
    """Yatay düzleme göre eğimi hesaplar."""
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle_rad = math.atan2(delta_y, delta_x)
    return math.degrees(angle_rad)

# --- ANA UYGULAMA ---
st.title("🧬 Pro Aesthetix: Kapsamlı Yüz Analizi")
st.markdown("Bu uygulama **20+ farklı metrik** ile ön yüz analizi ve özel **yan profil analizi** sunar.")

tab1, tab2 = st.tabs(["Ön Profil Analizi", "Yan Profil Analizi"])

# ==========================================
# TAB 1: ÖN PROFİL ANALİZİ
# ==========================================
with tab1:
    st.header("Ön Yüz Analizi")
    front_file = st.file_uploader("Önden çekilmiş fotoğraf yükleyin", type=["jpg", "png", "jpeg"], key="front")

    if front_file:
        image = Image.open(front_file)
        img_array = np.array(image)
        results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = img_array.shape
            
            # Koordinatları Kolay Almak İçin Lambda
            get_pt = lambda idx: (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

            # --- NOKTALAR ---
            # Gözler
            left_iris = get_pt(468)
            right_iris = get_pt(473)
            l_eye_in, l_eye_out = get_pt(362), get_pt(263)
            r_eye_in, r_eye_out = get_pt(133), get_pt(33)
            
            # Yüz Çerçevesi
            zygo_l, zygo_r = get_pt(234), get_pt(454) # Elmacıklar
            gonion_l, gonion_r = get_pt(58), get_pt(288) # Çene köşeleri
            menton = get_pt(152) # Çene ucu
            trichion = get_pt(10) # Saç çizgisi (yaklaşık)
            glabella = get_pt(9) # Kaş ortası

            # Burun & Ağız
            nose_tip = get_pt(1)
            nose_top = get_pt(168)
            alar_l, alar_r = get_pt(235), get_pt(456) # Burun kanatları
            mouth_l, mouth_r = get_pt(61), get_pt(291)
            lip_top, lip_bot = get_pt(0), get_pt(17)
            philtrum_top = get_pt(164)

            # Kaşlar
            brow_l_in, brow_l_out = get_pt(55), get_pt(46)
            brow_r_in, brow_r_out = get_pt(285), get_pt(276)

            # --- GÖRSELLEŞTİRME ---
            viz_img = img_array.copy()
            # Temel hatlar
            cv2.line(viz_img, zygo_l, zygo_r, (255, 0, 0), 2)
            cv2.line(viz_img, gonion_l, gonion_r, (255, 255, 0), 2)
            cv2.line(viz_img, trichion, menton, (0, 0, 255), 2)
            cv2.line(viz_img, l_eye_out, r_eye_out, (0, 255, 0), 2)
            st.image(viz_img, caption="Analiz Edilen Noktalar", use_container_width=True)

            # --- 20+ METRİK HESAPLAMA ---
            
            # Mesafeler
            face_width = calculate_distance(zygo_l, zygo_r)
            face_height = calculate_distance(trichion, menton)
            jaw_width = calculate_distance(gonion_l, gonion_r)
            eye_width_l = calculate_distance(l_eye_in, l_eye_out)
            eye_width_r = calculate_distance(r_eye_in, r_eye_out)
            ipd = calculate_distance(left_iris, right_iris) # Göz bebekleri arası
            nose_width = calculate_distance(alar_l, alar_r)
            mouth_width = calculate_distance(mouth_l, mouth_r)
            
            # Yüz Üçlüsü (Thirds)
            h_upper = calculate_distance(trichion, glabella)
            h_mid = calculate_distance(glabella, nose_tip) # Basitleştirilmiş
            h_lower = calculate_distance(nose_tip, menton)

            metrics = {
                "Canthal Tilt (Sol)": -calculate_tilt(l_eye_in, l_eye_out),
                "Canthal Tilt (Sağ)": calculate_tilt(r_eye_in, r_eye_out),
                "FWHR (Genişlik/Yükseklik)": face_width / face_height,
                "Midface Ratio (Kompaktlık)": calculate_distance(left_iris, right_iris) / face_width,
                "Çene/Elmacık Oranı": jaw_width / face_width,
                "Göz Açıklık Oranı (ESR)": calculate_distance(r_eye_in, l_eye_in) / ((eye_width_l + eye_width_r)/2),
                "Burun/Dudak Genişlik Oranı": nose_width / mouth_width,
                "Üst Dudak/Alt Dudak Oranı": calculate_distance(lip_top, mouth_l) / calculate_distance(lip_bot, mouth_l), # Yaklaşık kalınlık
                "Philtrum/Çene Oranı": calculate_distance(philtrum_top, lip_top) / calculate_distance(lip_bot, menton),
                "Yüz Üst 1/3 (%)": (h_upper / face_height) * 100,
                "Yüz Orta 1/3 (%)": (h_mid / face_height) * 100,
                "Yüz Alt 1/3 (%)": (h_lower / face_height) * 100,
                "Kaş Eğimi (Sol)": -calculate_tilt(brow_l_in, brow_l_out),
                "Kaş Eğimi (Sağ)": calculate_tilt(brow_r_in, brow_r_out),
                "Göz Boyutu Oranı": (eye_width_l + eye_width_r) / face_width,
                "Çene Ucu Genişliği Oranı": calculate_distance(get_pt(148), get_pt(377)) / mouth_width,
                "Yanak Dolgunluğu (Lower Cheek)": calculate_distance(gonion_l, mouth_l) / jaw_width,
                "Alın Genişliği Oranı": calculate_distance(get_pt(103), get_pt(332)) / face_width,
                "Burun Uzunluk Oranı": calculate_distance(glabella, nose_tip) / face_height,
                "Ağız Köşesi Yüksekliği": calculate_tilt(mouth_l, mouth_r) # Gülümseme eğimi
            }

            st.subheader("📊 20 Noktalı Detaylı Analiz Raporu")
            
            # Tablo oluşturma
            df_front = pd.DataFrame(list(metrics.items()), columns=["Metrik", "Değer"])
            st.dataframe(df_front, height=600, use_container_width=True)
            
            st.success("İdeal Oran Notları: FWHR 1.7-2.0 arası maskülen kabul edilir. Altın oranda yüz üçlüleri (Üst/Orta/Alt) %33.3 eşit olmalıdır.")

        else:
            st.error("Yüz bulunamadı.")

# ==========================================
# TAB 2: YAN PROFİL ANALİZİ
# ==========================================
with tab2:
    st.header("Yan Profil (Side Profile) Analizi")
    st.info("Lütfen başınızın tam yandan göründüğü bir fotoğraf yükleyin (Sağa veya Sola bakabilir).")
    
    side_file = st.file_uploader("Yan profil fotoğrafı yükleyin", type=["jpg", "png", "jpeg"], key="side")

    if side_file:
        image_side = Image.open(side_file)
        img_array_side = np.array(image_side)
        results_side = face_mesh.process(cv2.cvtColor(img_array_side, cv2.COLOR_RGB2BGR))

        if results_side.multi_face_landmarks:
            landmarks = results_side.multi_face_landmarks[0].landmark
            h, w, _ = img_array_side.shape
            get_pt_s = lambda idx: (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

            # Yan Profil İçin Kritik Noktalar
            # Not: MediaPipe 3D'dir ama yan profilde landmarklar kayabilir. En belirginleri seçiyoruz.
            # Profil sağa bakıyorsa ve sola bakıyorsa mantığı otomatik algılanmalı veya manuel seçilmeli.
            # Burada basitlik adına genel orta hat ve çene hattı noktalarını kullanacağız.
            
            nasion_s = get_pt_s(168) # Burun kökü
            pronasale = get_pt_s(1) # Burun ucu
            subnasale = get_pt_s(164) # Burun altı
            labrale_sup = get_pt_s(0) # Üst dudak
            labrale_inf = get_pt_s(17) # Alt dudak
            pogonion = get_pt_s(152) # Çene ucu
            gonion_s = get_pt_s(132) # Çene köşesi (Sağ taraf varsayılan, gerekirse sol 361)
            
            # Görselleştirme
            viz_side = img_array_side.copy()
            cv2.line(viz_side, nasion_s, pogonion, (255, 0, 0), 2) # Facial Plane
            cv2.line(viz_side, gonion_s, pogonion, (0, 255, 0), 2) # Mandibular Plane
            cv2.line(viz_side, subnasale, labrale_sup, (0, 0, 255), 2) # Nasolabial
            
            st.image(viz_side, caption="Yan Profil İşaretleri", use_container_width=True)

            # --- YAN PROFİL METRİKLERİ ---
            
            # 1. Gonial Angle (Çene Açısı) - Çok kritiktir. 
            # Kulak altı (yaklaşık) -> Gonion -> Pogonion
            # Kulak noktası MP'de tam yok, 132 (Gonion) ve 234 (Zygoma) ile dikey hat referansı alacağız.
            # Basit geometri: Çene hattının yatayla yaptığı açıya bakalım.
            jaw_angle = calculate_tilt(gonion_s, pogonion)
            
            # 2. Nasolabial Angle (Burun-Dudak Açısı)
            nasolabial_angle = calculate_angle(pronasale, subnasale, labrale_sup)
            
            # 3. Facial Convexity (Yüz Dışbükeyliği)
            # Glabella -> Subnasale -> Pogonion
            glabella_s = get_pt_s(9)
            convexity_angle = calculate_angle(glabella_s, subnasale, pogonion)

            # 4. Chin Projection (Çene Çıkıklığı)
            # Burun kökünden inen dikmeye göre çene nerede?
            # Pozitif değer çene ileride, negatif geride.
            chin_proj = pogonion[0] - nasion_s[0] # Basit pixel farkı (Yönü fotoğrafa göre değişir)

            side_metrics = {
                "Gonial Angle (Çene Açısı)": f"{abs(jaw_angle):.1f}° (Yatayla)",
                "Nasolabial Angle (Burun-Dudak)": f"{nasolabial_angle:.1f}°",
                "Yüz Konveksliği (Convexity)": f"{convexity_angle:.1f}°",
                "Çene Projeksiyonu": "İleri" if chin_proj > 0 else "Geri",
                "Burun Ucu Açısı": f"{calculate_angle(nasion_s, pronasale, subnasale):.1f}°"
            }
            
            st.subheader("📐 Yan Profil Ölçümleri")
            st.table(pd.DataFrame(list(side_metrics.items()), columns=["Özellik", "Değer"]))
            
            st.info("""
            **Bilgi:**
            * **Nasolabial Açı:** Erkeklerde 90-95°, Kadınlarda 95-100° ideal kabul edilir.
            * **Gonial Açı:** Keskin ve tanımlı bir çene hattı estetik bulunur.
            """)

        else:
            st.error("Yan profilde yüz algılanamadı.")

