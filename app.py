import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="TrafoGuard AI", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .main {background-color: #0E1117;}
    h1 {color: #FFD700;}
    .metric-card {background-color: #1E1E1E; border-left: 5px solid #FFD700; padding: 15px; margin-bottom: 10px; border-radius: 5px;}
    .status-good {color: #2ECC71; font-weight: bold; font-size: 24px;}
    .status-fair {color: #F1C40F; font-weight: bold; font-size: 24px;}
    .status-bad {color: #E74C3C; font-weight: bold; font-size: 24px;}
    .footer {position: fixed; bottom: 0; left: 0; width: 100%; background-color: #161B22; color: #888; text-align: center; padding: 10px; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

def process_image(image):
    img_resized = cv2.resize(image, (150, 150))
    x1, y1, x2, y2 = 40, 40, 110, 110
    cropped_img = img_resized[y1:y2, x1:x2]
    img_with_box = img_resized.copy()
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
    avg_per_row = np.average(cropped_img, axis=0)
    avg_color = np.average(avg_per_row, axis=0)
    return img_with_box, cropped_img, int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

def analyze_quality(R, G, B):
    if R == 0: R = 1
    ratio = G / R
    if ratio >= 0.75: return "BAIK (GOOD)", "Minyak Isolasi Prima", "status-good", 95, ratio
    elif 0.40 <= ratio < 0.75: return "CUKUP (FAIR)", "Indikasi Oksidasi Ringan", "status-fair", 60, ratio
    else: return "BURUK (POOR)", "Kritis! Segera Ganti", "status-bad", 25, ratio

st.title("⚡ TrafoGuard AI")
st.write("Sistem Deteksi Dini Degradasi Minyak Transformator")

uploaded_file = st.sidebar.file_uploader("Upload Citra Sampel", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    
    img_visual, img_crop, R, G, B = process_image(image)
    status, desc, css, score, ratio = analyze_quality(R, G, B)
    
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: st.image(img_visual, caption="Deteksi Area (ROI)", use_column_width=True)
    with c2: 
        st.image(img_crop, caption="Crop Analisis", width=150)
        st.color_picker("Warna Terdeteksi", f"#{R:02x}{G:02x}{B:02x}", disabled=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="{css}">{status}</div><div>{desc}</div></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={'text': "Health Score"}, gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 40], 'color': "#E74C3C"}, {'range': [40, 75], 'color': "#F1C40F"}, {'range': [75, 100], 'color': "#2ECC71"}]}))
        fig.update_layout(height=250, margin=dict(t=30,b=10,l=20,r=20), paper_bgcolor="#0E1117", font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Analisis Data")
    tab1, tab2 = st.tabs(["Rasio Warna", "Histogram"])
    with tab1:
        st.info(f"Rasio G/R Terhitung: **{ratio:.3f}** (Standar IEEE: > 0.75 Baik)")
        df = pd.DataFrame({'Channel': ['R', 'G', 'B'], 'Val': [R, G, B], 'C': ['#F00', '#0F0', '#00F']})
        st.bar_chart(df, x='Channel', y='Val', color='C')
    with tab2:
        colors = ('r', 'g', 'b')
        chart_data = pd.DataFrame()
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img_crop], [i], None, [256], [0, 256])
            chart_data[col] = hist.flatten()
        st.line_chart(chart_data)

st.markdown('<div class="footer">TrafoGuard AI Project | © 2026 Ahmaddani (23081015)</div>', unsafe_allow_html=True)
