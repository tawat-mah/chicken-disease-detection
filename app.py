import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
import os

st.set_page_config(
    page_title="ChickenAI - Disease Detection",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }

.stApp { background: #f5f5f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; padding: 0 !important; }
.stFileUploader > div { background: transparent !important; border: none !important; }
.stFileUploader label { display: none !important; }

/* TOPBAR */
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #e8e8e4;
    padding: 0 36px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.logo {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1a1a;
}
.logo-dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
}
.top-right {
    font-size: 0.72rem;
    color: #aaa;
    font-family: 'DM Mono', monospace;
}

/* MAIN GRID */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    padding: 24px 32px;
    height: calc(100vh - 56px);
    overflow: hidden;
}

/* CARD BASE */
.card {
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #e8e8e4;
    overflow: hidden;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.card-header {
    padding: 18px 20px 14px;
    border-bottom: 1px solid #f0f0ec;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #888;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.card-body {
    padding: 16px 20px;
    flex: 1;
    overflow-y: auto;
}

/* UPLOAD ZONE */
.upload-hint {
    border: 1.5px dashed #d4d4cc;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    background: #fafaf8;
    margin-bottom: 12px;
}
.upload-hint-text { font-size: 0.82rem; color: #aaa; margin-top: 6px; }

/* RESULT */
.result-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 10px;
    color: #ccc;
    text-align: center;
}
.result-empty-icon { font-size: 2.5rem; opacity: 0.4; }
.result-empty-text { font-size: 0.82rem; }

.detected-label { font-size: 0.65rem; color: #aaa; letter-spacing: 1px; text-transform: uppercase; }
.detected-disease { font-size: 1.5rem; font-weight: 700; color: #1a1a1a; margin: 6px 0 4px; }
.severity-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.result-desc { font-size: 0.78rem; color: #888; line-height: 1.6; margin-top: 10px; }

/* CONFIDENCE */
.conf-item { margin: 10px 0; }
.conf-row { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 4px; }
.conf-name { color: #555; }
.conf-pct { font-family: 'DM Mono', monospace; font-size: 0.72rem; }
.conf-track { background: #f0f0ec; border-radius: 3px; height: 5px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 3px; }

/* TAGS */
.section-mini { font-size: 0.65rem; font-weight: 600; color: #bbb; letter-spacing: 1px; text-transform: uppercase; margin: 14px 0 8px; }
.tag-row { display: flex; flex-wrap: wrap; gap: 5px; }
.tag {
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 6px;
    line-height: 1.5;
}
.tag-s { background: #fef3f0; color: #dc4e2a; }
.tag-t { background: #f0faf3; color: #1a7a3a; }
.tag-p { background: #f0f4ff; color: #3455cc; }

/* WARNING */
.warn-note {
    margin-top: 14px;
    padding: 10px 12px;
    background: #fffbf0;
    border: 1px solid #f0e0a0;
    border-radius: 8px;
    font-size: 0.7rem;
    color: #9a7a30;
    line-height: 1.5;
}

/* Streamlit button */
.stButton > button {
    background: #1a1a1a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover { background: #333 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #e0e0d8; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ===== Disease Data =====
DISEASES = {
    "Coccidiosis": {
        "color": "#ef4444", "sev": "MODERATE – SEVERE", "sev_bg": "#fef2f2", "sev_c": "#ef4444",
        "acc": "95%",
        "desc": "Parasitic infection (Eimeria) in the intestines. Common in young chickens.",
        "symptoms": ["Bloody/brown droppings","Lethargy, no appetite","Ruffled feathers","Diarrhea with mucus","Rapid weight loss"],
        "treatment": ["Amprolium in water","Toltrazuril (Baycox)","Sulfonamide antibiotics","Vitamin A & K supplements","Isolate sick birds"],
        "prevention": ["Vaccinate against Coccidiosis","Keep litter dry & clean","Avoid overcrowding","Preventive medication in feed","Replace litter regularly"]
    },
    "Healthy": {
        "color": "#22c55e", "sev": "NO DISEASE", "sev_bg": "#f0fdf4", "sev_c": "#16a34a",
        "acc": "79%",
        "desc": "Chicken is in good health. Droppings appear normal.",
        "symptoms": ["Brown or green droppings","White urate present","Firm well-formed stool","No abnormal odor"],
        "treatment": ["No treatment needed","Routine health care","Clean food & water","Regular monitoring"],
        "prevention": ["Balanced nutrition","Fresh water always","Clean ventilated housing","Scheduled vaccinations","Annual health check"]
    },
    "New Castle Disease": {
        "color": "#f97316", "sev": "CRITICAL", "sev_bg": "#fff7ed", "sev_c": "#ea580c",
        "acc": "86%",
        "desc": "Highly contagious viral disease. Can wipe out an entire flock rapidly.",
        "symptoms": ["Greenish watery droppings","Breathing difficulty","Severe lethargy","Twisted neck","Egg production drops"],
        "treatment": ["No direct antiviral cure","Symptomatic treatment","Antibiotics for secondary infections","Vitamins & electrolytes","Contact vet immediately!"],
        "prevention": ["Regular NCD vaccination","Prevent wild bird access","Disinfect housing & equipment","Control farm access","Quarantine new birds 2–3 wks"]
    },
    "Salmonella": {
        "color": "#eab308", "sev": "MODERATE", "sev_bg": "#fefce8", "sev_c": "#ca8a04",
        "acc": "92%",
        "desc": "Bacterial infection dangerous to both chickens and humans.",
        "symptoms": ["White or yellow droppings","Severe diarrhea","Weakness, lethargy","Loss of appetite","High chick mortality"],
        "treatment": ["Enrofloxacin antibiotic","Gentamicin injection","Trimethoprim-Sulfa","Electrolyte supplements","Consult vet first"],
        "prevention": ["Source chicks from certified farms","Disinfect feed & water","Wash hands after handling","Eliminate rodents","Regular cleaning"]
    }
}

@st.cache_resource
def load_model_and_encoder():
    import gdown, os, tempfile
    try:
        # ถ้ามีไฟล์ในเครื่องใช้เลย
        if os.path.exists('model/chicken_disease_model_efficientnetb0_final.h5'):
            from keras.models import load_model
            m = load_model('model/chicken_disease_model_efficientnetb0_final.h5')
            with open('model/label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            return m, le, True

        # โหลดจาก Google Drive
        os.makedirs('model', exist_ok=True)
        with st.spinner("⏳ Loading model from Google Drive (first time only)..."):
            MODEL_ID = "1Wf6KgOzRu5NKjF1a4PSbMhOgXtRKLvTR"
            ENCODER_ID = "1hqSf9qHw_RgW_LcQTsBX_RNzzj541iGj"
            gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}",
                          "model/chicken_disease_model_efficientnetb0_final.h5", quiet=False)
            gdown.download(f"https://drive.google.com/uc?id={ENCODER_ID}",
                          "model/label_encoder.pkl", quiet=False)

        from keras.models import load_model
        m = load_model('model/chicken_disease_model_efficientnetb0_final.h5')
        with open('model/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return m, le, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False

def predict(image, model, le):
    from keras.applications.efficientnet import preprocess_input
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224)).astype('float32')
    img = preprocess_input(img)
    pred = model.predict(np.expand_dims(img, 0), verbose=0)
    idx = np.argmax(pred[0])
    cls = le.inverse_transform([idx])[0]
    conf = pred[0][idx] * 100
    all_c = {le.inverse_transform([i])[0]: pred[0][i]*100 for i in range(len(le.classes_))}
    return cls, conf, all_c

model, le, ok = load_model_and_encoder()

# ===== TOPBAR =====
st.markdown("""
<div class="topbar">
    <div class="logo">
        <div class="logo-dot"></div>
        ChickenAI &nbsp;<span style="color:#ccc; font-weight:300;">Disease Detection</span>
    </div>
    <div class="top-right">EfficientNetB0 &nbsp;·&nbsp; Accuracy 90.3% &nbsp;·&nbsp; Designed by Tawat</div>
</div>
""", unsafe_allow_html=True)

# ===== 3 COLUMNS =====
c1, c2, c3 = st.columns([1, 1.1, 1], gap="small")

# ===== COL 1: UPLOAD =====
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span class="card-title">📁 &nbsp;Upload Image</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)

    if not ok:
        st.error("Model not found.")
    else:
        st.markdown("""
        <div style="border:1.5px dashed #d4d4cc; border-radius:8px; padding:8px 14px; background:#fafaf8; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
            <span style="font-size:1rem;">🔬</span>
            <span style="font-size:0.75rem; color:#bbb;">JPG · PNG · Max 10MB</span>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("upload", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if uploaded:
            image = Image.open(uploaded)
            if st.button("⚡  Run Diagnosis"):
                with st.spinner("Analyzing..."):
                    disease, conf, all_c = predict(image, model, le)
                    st.session_state['result'] = {'disease': disease, 'conf': conf, 'all_c': all_c}
                st.rerun()
            st.image(image, use_container_width=True, caption=uploaded.name)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ===== COL 2: RESULT =====
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span class="card-title">🩺 &nbsp;Diagnosis Result</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)

    if 'result' in st.session_state and ok:
        r = st.session_state['result']
        d = r['disease']
        info = DISEASES[d]

        st.markdown(f"""
        <div class="detected-label">DETECTED</div>
        <div class="detected-disease">{d}</div>
        <span class="severity-badge" style="background:{info['sev_bg']}; color:{info['sev_c']};">{info['sev']}</span>
        <div class="result-desc">{info['desc']}</div>
        <div class="section-mini" style="margin-top:20px;">Confidence Scores</div>
        """, unsafe_allow_html=True)

        for cls, pct in sorted(r['all_c'].items(), key=lambda x: -x[1]):
            ci = DISEASES[cls]
            bold = "font-weight:600; color:#1a1a1a;" if cls == d else "color:#aaa;"
            name = cls if cls != "New Castle Disease" else "Newcastle Disease"
            st.markdown(f"""
            <div class="conf-item">
                <div class="conf-row">
                    <span class="conf-name" style="{bold}">{name}</span>
                    <span class="conf-pct" style="color:{ci['color'] if cls==d else '#ccc'}">{pct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{pct:.1f}%; background:{ci['color']};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warn-note">
            ⚠️ This result is for preliminary assessment only.
            Always consult a licensed veterinarian before treatment.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-empty">
            <div class="result-empty-icon">🔬</div>
            <div class="result-empty-text">Upload an image and<br>run diagnosis to see results</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ===== COL 3: INFO =====
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span class="card-title">📋 &nbsp;Disease Information</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)

    if 'result' in st.session_state and ok:
        d = st.session_state['result']['disease']
        info = DISEASES[d]

        st.markdown('<div class="section-mini">Symptoms</div><div class="tag-row">', unsafe_allow_html=True)
        for s in info['symptoms']:
            st.markdown(f'<span class="tag tag-s">{s}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if d != "Healthy":
            st.markdown('<div class="section-mini">Treatment</div><div class="tag-row">', unsafe_allow_html=True)
            for t in info['treatment']:
                st.markdown(f'<span class="tag tag-t">{t}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-mini">Prevention</div><div class="tag-row">', unsafe_allow_html=True)
        for p in info['prevention']:
            st.markdown(f'<span class="tag tag-p">{p}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-empty">
            <div class="result-empty-icon">📋</div>
            <div class="result-empty-text">Disease details will<br>appear here after diagnosis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)
