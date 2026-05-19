import streamlit as st
import joblib
import re
import unicodedata
import pandas as pd
from googleapiclient.discovery import build

@st.cache_resource
def load_ai():
    vectorizer = joblib.load('judol_vectorizer.pkl')
    model = joblib.load('judol_model.pkl')
    return vectorizer, model

vectorizer, model = load_ai()

def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_youtube_comments(api_key, video_id, max_results=None):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments_data = []
        next_page_token = None
        
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comments_data.append({'Username': author, 'Komentar': comment})
                
            next_page_token = response.get('nextPageToken')
            
            if max_results and len(comments_data) >= max_results:
                comments_data = comments_data[:max_results]
                break
                
            if not next_page_token:
                break
                
        return comments_data
    except Exception as e:
        st.error(f"Gagal mengambil komentar. Pastikan API Key dan URL valid. Error: {e}")
        return None

st.set_page_config(page_title="Deteksi Judi Online", page_icon="🚨", layout="wide")
st.title("🚨 Deteksi Komentar Promosi Judi Online")
st.write("Aplikasi ini menggunakan Machine Learning untuk menyaring komentar spam/promosi judi online di YouTube.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Masukkan YouTube API Key:", type="password", help="Dapatkan di Google Cloud Console")
    st.info("API Key diperlukan untuk menarik komentar langsung dari link YouTube.")

tab1, tab2 = st.tabs(["🎥 Cek via Link YouTube", "✍️ Cek Manual"])

with tab1:
    youtube_url = st.text_input("Masukkan Link Video YouTube:", placeholder="Contoh: https://www.youtube.com/watch?v=xxxxxxxxx")
    
    fetch_all = st.checkbox("Tarik SEMUA komentar (Bisa memakan waktu lama untuk video yang sangat ramai)")
    
    max_komentar = None
    if not fetch_all:
        max_komentar = st.slider("Batasi jumlah komentar yang ditarik:", 10, 500, 100, step=10)
    
    if st.button("Tarik & Cek Komentar", type="primary"):
        if not api_key:
            st.warning("⚠️ Masukkan YouTube API Key terlebih dahulu di bilah kiri (sidebar)!")
        elif not youtube_url:
            st.warning("⚠️ Link YouTube tidak boleh kosong!")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("❌ Format link YouTube tidak valid!")
            else:
                with st.spinner("Mengambil komentar dan menganalisis... (Mohon tunggu)"):
                    comments_list = get_youtube_comments(api_key, video_id, max_results=max_komentar)
                    
                    if comments_list:
                        df = pd.DataFrame(comments_list)
                        df['Clean_Text'] = df['Komentar'].apply(clean_text)
                        
                        X_input = vectorizer.transform(df['Clean_Text'])
                        probabilities = model.predict_proba(X_input)[:, 1]
                        threshold = 0.85 
                        predictions = (probabilities >= threshold).astype(int)

                        df['Prediksi'] = predictions
                        
                        df_judol = df[df['Prediksi'] == 1]
                        df_aman = df[df['Prediksi'] == 0]
                        
                        st.subheader(f"Hasil Analisis ({len(comments_list)} komentar berhasil ditarik):")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.error(f"🚨 Terdeteksi Judol: {len(df_judol)} komentar")
                            if not df_judol.empty:
                                st.dataframe(df_judol[['Username', 'Komentar']], use_container_width=True)
                            else:
                                st.success("Tidak ada komentar judi online yang ditemukan.")
                                
                        with col2:
                            st.success(f"✅ Komentar Aman: {len(df_aman)} komentar")
                            if not df_aman.empty:
                                st.dataframe(df_aman[['Username', 'Komentar']], use_container_width=True)

with tab2:
    user_input = st.text_area("Masukkan teks komentar di sini:", placeholder="Contoh: Ayo main di bazartoto pasti gacor...")
    if st.button("Cek Komentar Manual"):
        if user_input.strip() == "":
            st.warning("Komentar tidak boleh kosong!")
        else:
            cleaned_input = clean_text(user_input)
            vec_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vec_input)
            
            st.divider()
            if prediction[0] == 1:
                st.error("🚨 TERDETEKSI PROMOSI JUDI ONLINE!")
                st.write("Komentar ini sangat mirip dengan pola spammer/promotor judi.")
            else:
                st.success("✅ KOMENTAR AMAN.")
                st.write("Komentar ini terdeteksi sebagai komentar normal.")