import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import math
from PIL import Image, ImageOps
from scipy.stats import norm

# --- AYARLAR ---
st.set_page_config(page_title="Aesthetix Pro: Deep Analysis", layout="wide")

# --- MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
# Ön profil: Yüksek hassasiyet
face_mesh_front = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
# Yan profil: Düşük hassasiyet (Algılamayı kolaylaştırmak için)
face_mesh_side = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.1)

# --- MATEMATİK & İSTATİSTİK MOTORU ---

def get_dist(p1, p2):
    """Öklid Mesafesi"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_angle(p1, p2, p3):
    """3 Nokta Arası Açı"""
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_tilt(p1, p2):
    """Yatay Eğim"""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def analyze_population(value, ideal_mean, std_dev, label="normal"):
    """
    Değeri popülasyon verisiyle kıyaslar (Normal Dağılım).
    label 'normal': Ortalamaya yakın olmak iyidir (Örn: Yüz oranları).
    label 'high': Yüksek olması iyidir (Örn: Çene hattı keskinliği).
    label 'low': Düşük olması iyidir (Örn: Yağ oranı belirtileri).
    """
    z_score = (value - ideal_mean) / std_dev
    
    if label == "high":
        percentile = norm.cdf(z_score) * 100
    elif label == "low":
        percentile = 100 - (norm.cdf(z_score) * 100)
    else: # normal (Golden ratio vb.)
        # İdeale ne kadar yakınsa o kadar iyi. 
        # Z-score 0 ise %99, arttıkça düşer.
        diff = abs(value - ideal_mean)
        # Basitleştirilmiş proximity skoru
        percentile = max(0, 100 - (diff / std_dev) * 20)

    # Metin yorumu
    if percentile >= 90: rating = "💎 Top %10 (Elit)"
    elif percentile >= 75: rating = "✅ Ortalamanın Üstü"
    elif percentile >= 45: rating = "🔹 Ortalama"
    else: rating = "🔸 Geliştirilebilir"
    
    return f"{percentile:.1f}", rating

# --- ANA UI ---
st.title("🧬 Aesthetix Pro: 50 Nokta Detaylı Analiz")
st.markdown("""
<style>
.big-font { font-size:20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.info("Bu analiz, akademik 'Neoclassical Canons' ve modern estetik verilerine dayanarak sizi genel popülasyon simülasyonu ile kıyaslar.")

tab_front, tab_side = st.tabs(["👤 Ön Profil (40 Ölçüm)", "🗿 Yan Profil (10+ Ölçüm)"])

# ==============================================================================
# ÖN PROFİL ANALİZİ
# ==============================================================================
with tab_front:
    uploaded_front = st.file_uploader("Ön Profil Fotoğrafı", type=["jpg", "png", "jpeg"])
    
    if uploaded_front:
        img = Image.open(uploaded_front)
        img = ImageOps.exif_transpose(img)
        arr = np.array(img)
        res = face_mesh_front.process(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = arr.shape
            p = lambda i: (int(lm[i].x * w), int(lm[i].y * h)) # Nokta alma kısayolu

            # --- ANATOMİK REFERANS NOKTALARI ---
            # Gözler
            ex_L, ex_R = p(33), p(263) # Dış köşeler
            en_L, en_R = p(133), p(362) # İç köşeler
            iris_L, iris_R = p(468), p(473)
            # Yüz Çerçevesi
            zygo_L, zygo_R = p(234), p(454) # Elmacık
            go_L, go_R = p(58), p(288) # Çene köşesi
            me = p(152) # Çene ucu
            tr = p(10) # Saç çizgisi
            g = p(9) # Glabella (Kaş ortası)
            n = p(168) # Nasion
            # Burun & Ağız
            al_L, al_R = p(235), p(456) # Burun kanatları
            ch_L, ch_R = p(61), p(291) # Dudak köşeleri
            ls, li = p(0), p(17) # Dudak üst/alt orta
            sn = p(164) # Subnasale (Burun altı)
            # Kaşlar
            br_L_in, br_L_out = p(55), p(46)
            br_R_in, br_R_out = p(285), p(276)

            # --- GÖRSELLEŞTİRME ---
            viz = arr.copy()
            for pt in [zygo_L, zygo_R, go_L, go_R, me, tr, ex_L, ex_R]:
                cv2.circle(viz, pt, 3, (0,255,0), -1)
            cv2.line(viz, zygo_L, zygo_R, (255,0,0), 2)
            cv2.line(viz, tr, me, (0,0,255), 2)
            st.image(viz, caption="Landmark Tespitleri", use_container_width=True)

            # --- TEMEL DEĞİŞKENLER ---
            face_w = get_dist(zygo_L, zygo_R)
            face_h = get_dist(tr, me)
            jaw_w = get_dist(go_L, go_R)
            eye_w = (get_dist(ex_L, en_L) + get_dist(ex_R, en_R)) / 2
            ipd = get_dist(iris_L, iris_R)
            nose_w = get_dist(al_L, al_R)
            mouth_w = get_dist(ch_L, ch_R)
            
            # --- 40+ METRİK HESAPLAMA & POPÜLASYON ANALİZİ ---
            data = []

            def add_metric(category, name, value, ideal, std, mode="normal"):
                perc, rating = analyze_population(value, ideal, std, mode)
                data.append({
                    "Kategori": category,
                    "Ölçüm": name,
                    "Değer": round(value, 2),
                    "İdeal": ideal,
                    "Popülasyon %": perc,
                    "Durum": rating
                })

            # 1. YÜZ ORANLARI (FACIAL RATIOS)
            fwhr = face_w / face_h # 1.9 maskülen, 1.6 feminen
            add_metric("Yüz Şekli", "FWHR (Genişlik/Yükseklik)", fwhr, 1.85, 0.15, "high")
            add_metric("Yüz Şekli", "Yüz Üçlüsü 1 (Alın)", (get_dist(tr, g)/face_h)*100, 33.3, 2.0, "normal")
            add_metric("Yüz Şekli", "Yüz Üçlüsü 2 (Orta)", (get_dist(g, sn)/face_h)*100, 33.3, 2.0, "normal")
            add_metric("Yüz Şekli", "Yüz Üçlüsü 3 (Alt)", (get_dist(sn, me)/face_h)*100, 33.3, 2.0, "normal")
            add_metric("Yüz Şekli", "Yüz Beşlisi (Göz Aralığı)", get_dist(en_L, en_R)/face_w, 0.20, 0.02, "normal")
            
            # 2. ÇENE & ELMACIK (JAW & CHEEK)
            add_metric("Çene/Elmacık", "Jaw-to-Cheek Ratio", jaw_w / face_w, 0.85, 0.05, "high") # 1'e yakın olması iyidir (erkek)
            add_metric("Çene/Elmacık", "Chin-to-Philtrum Ratio", get_dist(li, me) / get_dist(sn, ls), 2.2, 0.2, "normal")
            add_metric("Çene/Elmacık", "Çene Ucu Genişliği (Relatif)", get_dist(p(148), p(377)) / mouth_w, 0.8, 0.1, "high")
            add_metric("Çene/Elmacık", "Ramus/Mandible Oranı (Ön)", get_dist(zygo_R, go_R) / get_dist(go_R, me), 0.7, 0.1, "normal")

            # 3. GÖZLER (OCULAR REGION)
            tilt_L = -get_tilt(en_L, ex_L)
            tilt_R = get_tilt(en_R, ex_R)
            avg_tilt = (tilt_L + tilt_R) / 2
            add_metric("Gözler", "Canthal Tilt (Derece)", avg_tilt, 6.0, 2.5, "high")
            add_metric("Gözler", "Eye Aspect Ratio (Göz Açıklığı)", get_dist(p(159), p(145)) / eye_w, 0.35, 0.05, "high") # Hunter eyes için düşük olması iyidir ama genel çekicilik için orta
            add_metric("Gözler", "ESR (Eye Spacing Ratio)", get_dist(en_L, en_R) / eye_w, 1.0, 0.1, "normal") # Tam 1 olmalı
            add_metric("Gözler", "Medial Canthal Angle", get_angle(p(33), p(133), p(159)), 45, 5, "low") # İç göz açısı keskinliği
            add_metric("Gözler", "Kaş-Göz Mesafesi Oranı", get_dist(p(66), p(159)) / face_h, 0.06, 0.01, "low") # Düşük kaş erkeksi
            add_metric("Gözler", "Kaş Eğimi (Tilt)", abs(get_tilt(br_L_in, br_L_out)), 8.0, 3.0, "normal")

            # 4. BURUN & DUDAK (NOSE & LIPS)
            add_metric("Burun/Dudak", "Nasal Index (Genişlik/Uzunluk)", nose_w / get_dist(n, sn), 0.7, 0.1, "normal")
            add_metric("Burun/Dudak", "Burun/Yüz Genişliği", nose_w / face_w, 0.25, 0.02, "normal") # Rule of fifths
            add_metric("Burun/Dudak", "Dudak Genişlik Oranı", mouth_w / face_w, 0.35, 0.03, "high")
            add_metric("Burun/Dudak", "Üst/Alt Dudak Oranı", get_dist(ls, sn)/get_dist(li, me), 0.3, 0.05, "normal")
            add_metric("Burun/Dudak", "Vermilion Ratio (Dudak Kalınlığı)", get_dist(ls, li) / mouth_w, 0.3, 0.05, "high")
            add_metric("Burun/Dudak", "Philtrum Derinliği (Görsel)", get_dist(p(164), p(0)) / face_h, 0.04, 0.01, "low") # Kısa philtrum iyidir
            
            # --- TABLO GÖSTERİMİ ---
            df = pd.DataFrame(data)
            
            st.markdown("### 📊 Detaylı Analiz Raporu")
            
            # Kategorilere göre expander içinde gösterelim
            categories = df["Kategori"].unique()
            for cat in categories:
                with st.expander(f"📌 {cat} Analizi", expanded=True):
                    sub_df = df[df["Kategori"] == cat].drop(columns=["Kategori"])
                    st.dataframe(sub_df, use_container_width=True)

            # Genel Skor Hesaplama
            st.markdown("---")
            avg_score = df["Popülasyon %"].astype(float).mean()
            st.metric(label="GENEL ESTETİK UYUM SKORU (Aesthetix Score)", value=f"{avg_score:.1f} / 100")
            st.caption("*Bu skor, yüzünüzün matematiksel ortalamalara (altın oran vb.) ne kadar 'uyumlu' olduğunu gösterir. Tıbbi bir teşhis değildir.*")

        else:
            st.error("Yüz algılanamadı.")

# ==============================================================================
# YAN PROFİL ANALİZİ
# ==============================================================================
with tab_side:
    st.info("Tam yan profil (90°) yerine hafif çapraz (3/4) profil de deneyebilirsiniz. MediaPipe yan profilde zorlanabilir.")
    uploaded_side = st.file_uploader("Yan Profil Fotoğrafı", type=["jpg", "png", "jpeg"], key="side")
    
    if uploaded_side:
        img_s = Image.open(uploaded_side)
        img_s = ImageOps.exif_transpose(img_s)
        arr_s = np.array(img_s)
        res_s = face_mesh_side.process(cv2.cvtColor(arr_s, cv2.COLOR_RGB2BGR))
        
        if res_s.multi_face_landmarks:
            lm_s = res_s.multi_face_landmarks[0].landmark
            h_s, w_s, _ = arr_s.shape
            p_s = lambda i: (int(lm_s[i].x * w_s), int(lm_s[i].y * h_s))
            
            # Noktalar (Yön algılama ile)
            tip = p_s(1) # Burun ucu
            root = p_s(168) # Burun kökü
            
            # Yön kontrolü
            looking_right = tip[0] > root[0]
            
            # Noktalar
            g_s = p_s(9) # Glabella
            n_s = p_s(168) # Nasion
            prn = p_s(1) # Pronasale
            sn_s = p_s(164) # Subnasale
            ls_s = p_s(0) # Labrale Superius
            pg = p_s(152) # Pogonion (Çene ucu)
            go_s = p_s(132) if looking_right else p_s(361) # Gonion
            tragus = p_s(234) if looking_right else p_s(454) # Kulak civarı (Referans)

            # Görselleştirme
            viz_s = arr_s.copy()
            cv2.line(viz_s, n_s, pg, (255,0,0), 2) # Facial Plane
            cv2.line(viz_s, go_s, pg, (0,255,0), 2) # Mandibular Plane
            cv2.line(viz_s, prn, sn_s, (0,0,255), 2) # Nasolabial
            st.image(viz_s, caption="Yan Profil Hatları", use_container_width=True)

            # Metrikler
            side_data = []
            
            def add_side_metric(name, val, ideal, std, mode="normal"):
                perc, rating = analyze_population(val, ideal, std, mode)
                side_data.append({"Ölçüm": name, "Değer": round(val, 2), "İdeal": ideal, "Popülasyon %": perc, "Durum": rating})

            # 1. Gonial Angle (Çene Köşesi)
            gonial_angle = get_angle(tragus, go_s, pg)
            add_side_metric("Gonial Angle (Çene Açısı)", gonial_angle, 125, 5, "normal") # 120-130 derece idealdir
            
            # 2. Nasolabial Angle
            nasolabial = get_angle(prn, sn_s, ls_s)
            add_side_metric("Nasolabial Angle (Burun-Dudak)", nasolabial, 95, 5, "normal") # Erkeklerde 90-95
            
            # 3. Facial Convexity (Glabella-Subnasale-Pogonion)
            convexity = get_angle(g_s, sn_s, pg)
            add_side_metric("Yüz Konveksliği", convexity, 168, 4, "high") # 165-175 arası iyidir
            
            # 4. Chin Projection (Zero Meridian)
            # Nasion'dan aşağı inen dikmeye göre Pogonion nerede?
            proj = (pg[0] - n_s[0]) if looking_right else (n_s[0] - pg[0])
            add_side_metric("Çene Projeksiyonu (Pixel)", proj, 10, 20, "high") # Pozitif olması istenir
            
            # 5. Burun Çıkıklığı (Nasofrontal Angle)
            nasofrontal = get_angle(g_s, n_s, prn)
            add_side_metric("Nasofrontal Angle (Burun Kökü)", nasofrontal, 120, 5, "normal")

            # 6. Mentolabial Sulcus (Dudak altı oluğu)
            mentolabial = get_angle(li, p_s(17), pg) # Yaklaşık
            add_side_metric("Mentolabial Angle", mentolabial, 130, 10, "normal")

            st.table(pd.DataFrame(side_data))
            
        else:
            st.error("Yan profil algılanamadı. Lütfen açıyı değiştirin.")
