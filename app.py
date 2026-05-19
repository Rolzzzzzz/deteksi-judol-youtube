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

BLACKLIST_WORDS = [
    'banteng hoki', 'banteng', 
    'b a n t e n g',
    'b a z a r', 'bazartoto', 
    'zeus', 'maxwin', 'slot gacor'
]

SPAM_EMOJIS = ['⭐', '⚡', '🔥', '🤑', '❤']

def check_blacklist_and_emoji(raw_text, cleaned_text):
    for word in BLACKLIST_WORDS:
        if word in cleaned_text:
            return True, "Tertangkap Rule-Based: Mengandung kata kunci situs/brand judol mutlak."
            
    for emoji in SPAM_EMOJIS:
        if emoji in raw_text:
            if re.search(r'[A-Za-z]\s[A-Za-z]\s[A-Za-z]', raw_text):
                return True, f"Tertangkap Rule-Based: Menggunakan emoji spam ({emoji}) dipadu dengan trik manipulasi spasi."
                
    return False, ""

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
st.write("Aplikasi ini menggunakan Machine Learning & Rule-Based Filtering untuk menyaring komentar spam/promosi judi online di YouTube.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Masukkan YouTube API Key:", type="password", help="Dapatkan di Google Cloud Console")
    
    st.divider()
    st.subheader("🎛️ Sensitivitas AI")
    batas_keyakinan = st.slider("Batas Keyakinan (Threshold)", min_value=0.50, max_value=0.95, value=0.65, step=0.05)
    st.caption("Turunkan angka jika banyak komen judol yang lolos. Naikkan angka jika komentar normal ikut terdeteksi sebagai judol.")

tab1, tab2 = st.tabs(["🎥 Cek via Link YouTube", "✍️ Cek Manual"])

with tab1:
    youtube_url = st.text_input("Masukkan Link Video YouTube:", placeholder="Contoh: https://www.youtube.com/watch?v=xxxxxxxxx")
    fetch_all = st.checkbox("Tarik SEMUA komentar")
    
    max_komentar = None
    if not fetch_all:
        max_komentar = st.slider("Batasi jumlah komentar yang ditarik:", 10, 500, 100, step=10)
    
    if st.button("Tarik & Cek Komentar", type="primary"):
        if not api_key or not youtube_url:
            st.warning("⚠️ Masukkan API Key dan Link YouTube dengan benar!")
        else:
            video_id = extract_video_id(youtube_url)
            if video_id:
                with st.spinner("Mengambil komentar dan menganalisis..."):
                    comments_list = get_youtube_comments(api_key, video_id, max_results=max_komentar)
                    if comments_list:
                        df = pd.DataFrame(comments_list)
                        df['Clean_Text'] = df['Komentar'].apply(clean_text)
                        
                        prediksi_final = []
                        X_input = vectorizer.transform(df['Clean_Text'])
                        probabilities = model.predict_proba(X_input)[:, 1]
                        
                        for i, cleaned_str in enumerate(df['Clean_Text']):
                            raw_str = df['Komentar'].iloc[i]
                            is_spam, alasan = check_blacklist_and_emoji(raw_str, cleaned_str)
                            
                            if is_spam:
                                prediksi_final.append(1)
                            else:
                                prediksi_final.append(1 if probabilities[i] >= batas_keyakinan else 0)
                                
                        df['Prediksi'] = prediksi_final
                        
                        df_judol = df[df['Prediksi'] == 1]
                        df_aman = df[df['Prediksi'] == 0]
                        
                        st.subheader(f"Hasil Analisis ({len(comments_list)} komentar berhasil ditarik):")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.error(f"🚨 Terdeteksi Judol: {len(df_judol)} komentar")
                            if not df_judol.empty:
                                st.dataframe(df_judol[['Username', 'Komentar']], use_container_width=True)
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
            
            is_spam, alasan = check_blacklist_and_emoji(user_input, cleaned_input)
            
            if is_spam:
                st.divider()
                st.error("🚨 TERDETEKSI PROMOSI JUDI ONLINE!")
                st.write(alasan)
            else:
                vec_input = vectorizer.transform([cleaned_input])
                probabilities = model.predict_proba(vec_input)[:, 1]
                persentase = probabilities[0] * 100
                
                st.divider()
                if probabilities[0] >= batas_keyakinan:
                    st.error(f"🚨 TERDETEKSI PROMOSI JUDI ONLINE! (Keyakinan AI: {persentase:.1f}%)")
                else:
                    st.success(f"✅ KOMENTAR AMAN. (Keyakinan Judol AI hanya: {persentase:.1f}%)")