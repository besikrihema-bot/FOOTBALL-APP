import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Page Configuration ---
st.set_page_config(
    page_title="تحليل أسعار لاعبي كرة القدم | Player Valuation AI",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed" 
)

# --- Modern, Eye-Comfortable CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    
    /* Main Background: Soft Dark Blue */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        font-family: 'Cairo', sans-serif;
        direction: rtl;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-left: 1px solid #334155;
    }
    
    h1, h2, h3, h4, label, .stMarkdown {
        font-family: 'Cairo', sans-serif !important;
        color: #e2e8f0 !important;
    }
    
    h1 {
        text-align: center;
        background: linear-gradient(to right, #2dd4bf, #38bdf8); /* Teal to Sky Blue */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        padding-bottom: 20px;
    }
    
    /* Inputs & Selectboxes */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: #475569 !important;
        color: #f8fafc !important;
        border-radius: 10px;
    }
    
    .stNumberInput input, .stTextInput input {
        background-color: #1e293b !important;
        border-color: #475569 !important;
        color: #f8fafc !important;
        border-radius: 10px;
    }

    /* --- Modern Sliders (Teal/Clean) --- */
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #2dd4bf, #0ea5e9) !important; /* Gradient Track */
        height: 6px !important;
    }
    
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #f1f5f9 !important; /* White Handle for contrast */
        border: 2px solid #0ea5e9;
        box-shadow: 0 0 10px rgba(14, 165, 233, 0.3);
        width: 18px !important;
        height: 18px !important;
    }
    
    .stSlider label {
        color: #94a3b8 !important; /* Softer text for labels */
        font-weight: 600;
    }
    
    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2dd4bf 0%, #0ea5e9 100%);
        color: #0f172a !important;
        font-weight: 700 !important;
        border: none;
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.1rem !important;
        margin-top: 20px;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        transition: all 0.3s ease;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- ML Model (Cached) ---
@st.cache_resource
def get_model():
    # Synthetic Data Generation
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'age': np.random.randint(16, 40, n),
        'height': np.random.normal(180, 7, n),
        'league_coef': np.random.choice([2.0, 3.5, 4.5, 5.0], n),
        'rating': np.random.normal(70, 10, n),
        'matches_score': np.random.choice([1, 2, 3, 4], n), # 1=Low, 4=High
        'goals_score': np.random.choice([1, 2, 3, 4, 5], n),
        'fame_score': np.random.randint(1, 6, n),
        'discipline_score': np.random.randint(1, 11, n),
        'injury_coef': np.random.choice([0.6, 1.0], n)
    })
    
    # Target Calculation (Rule-based for training)
    def pricing(r):
        base = 50000 * pow(1.12, (r['rating']-50))
        age_f = 1.0 if r['age'] > 22 else 1.2
        perf_f = (r['goals_score']/3) * (r['matches_score']/3)
        return int(base * r['league_coef'] * age_f * perf_f * r['injury_coef'])
    
    df['value'] = df.apply(pricing, axis=1)
    
    # Train
    X = df.drop('value', axis=1)
    y = df['value']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = get_model()

# --- APP UI ---
st.markdown("<h1>تحليل أسعار لاعبي كرة القدم</h1>", unsafe_allow_html=True)

# 1. Personal & Physical (Grouped)
with st.container():
    st.markdown("### 👤 الملف الشخصي والبدني")
    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("العمر", 16, 45, 24)
    with c2: height = st.number_input("الطول (سم)", 150, 210, 180)
    with c3: weight = st.number_input("الوزن (كجم)", 50, 120, 75)
    
    c4, c5 = st.columns(2)
    with c4: nationality = st.selectbox("الجنسية", ["محلي", "أجنبي (أوروبا)", "أجنبي (أمريكا الجنوبية)", "أجنبي (أفريقيا/آسيا)"])
    with c5: foot = st.selectbox("القدم", ["اليمنى", "اليسرى", "كلتاهما"])

st.divider()

# 2. Football Info & Attributes (Sliders)
st.markdown("### ⚽ القدرات الفنية والبدنية")
col_tech, col_phys = st.columns(2)

with col_tech:
    st.caption("المهارات الأساسية")
    position = st.selectbox("المركز", ["مهاجم (ST)", "جناح (Winger)", "صانع لعب (CAM)", "وسط (CM)", "دفاع (CB)", "حارس (GK)"])
    skill = st.slider("المهارة / المراوغة", 0, 100, 75)
    passing = st.slider("دقة التمرير", 0, 100, 70)
    shooting = st.slider("إنهاء الهجمات", 0, 100, 70)

with col_phys:
    st.caption("اللياقة والقوة")
    speed = st.slider("السرعة / التسارع", 0, 100, 80)
    strength = st.slider("القوة الجسدية", 0, 100, 75)
    stamina = st.slider("معدل التحمل", 0, 100, 70)
    # Hidden calc items
    control, vision, agility = 70, 70, 70 

st.divider()

# 3. Context & Status (Dropdowns Only - No Sliders here)
st.markdown("### 📊 الأداء والحالة (Context)")
cc1, cc2 = st.columns(2)

with cc1:
    # League
    league_map = {"الدوري الممتاز (Top 5)": 5.0, "دوري درجة أولى قوى": 4.0, "دوري متوسط": 3.0, "دوري ضعيف": 2.0}
    league_sel = st.selectbox("مستوى الدوري الحالي", list(league_map.keys()))
    league_coef = league_map[league_sel]

    # Performance (Matches)
    matches_map = {"شارك في كل المباريات (+35)": 4, "لاعب أساسي (+25)": 3, "لاعب تدوير (15-25)": 2, "مشاركات قليلة (<15)": 1}
    matches_sel = st.selectbox("معدل المشاركة (الموسم الماضي)", list(matches_map.keys()))
    matches_score = matches_map[matches_sel]

    # Scoring/Assist
    goals_map = {"هـداف الدوري / صانع ألعاب سوبر": 5, "مساهمات عالية جداً": 4, "مساهمات جيدة": 3, "مساهمات عادية": 2, "قليلة / دفاعي": 1}
    goals_sel = st.selectbox("المساهمة التهديفية", list(goals_map.keys()))
    goals_score = goals_map[goals_sel]

with cc2:
    # Fame
    fame_map = {"نجم عالمي (Global Icon)": 5, "نجم قاري / دولي": 4, "نجم محلي مشهور": 3, "معروف في دوريه": 2, "مغمور / صاعد": 1}
    fame_sel = st.selectbox("الشهرة الجماهيرية", list(fame_map.keys()))
    fame_score = fame_map[fame_sel]

    # Discipline
    disc_map = {"مثالي (قائد في الملعب)": 10, "منضبط جداً": 8, "متوسط (بعض البطاقات)": 6, "مشاغب / بطاقات كثيرة": 3}
    disc_sel = st.selectbox("السلوك والانضباط", list(disc_map.keys()))
    disc_score = disc_map[disc_sel]

    # Injury
    inj_map = {"سليم (جاهز دائماً)": 1.0, "إصابات عضلية عادية": 0.9, "تاريخ إصابات مقلق": 0.7, "عائد من إصابة طويلة": 0.6}
    inj_sel = st.selectbox("الحالة الطبية", list(inj_map.keys()))
    injury_coef = inj_map[inj_sel]

# Action
if st.button("تحديث التقييم 💰"):
    with st.spinner("جاري المعالجة..."):
        time.sleep(0.5)
        
        # Calc Rating for model
        rating = (speed+strength+stamina + skill+cards+passing+shooting)/4.5 # Simplified avg
        if rating > 99: rating=99
        
        # Predict
        # Input vector: ['age', 'height', 'league_coef', 'rating', 'matches_score', 'goals_score', 'fame_score', 'discipline_score', 'injury_coef']
        x_in = pd.DataFrame([{
            'age': age, 'height': height, 'league_coef': league_coef,
            'rating': rating, 'matches_score': matches_score, 'goals_score': goals_score,
            'fame_score': fame_score, 'discipline_score': disc_score, 'injury_coef': injury_coef
        }])
        
        pred_val = model.predict(x_in)[0]
        final_val = int(pred_val * np.random.uniform(0.95, 1.05)) # variance
        
        # Display
        s_val = f"${final_val:,.0f}"
        
        # Badge
        badge = "💎" if final_val > 50000000 else "🔥" if final_val > 10000000 else "⚽"
        
        st.markdown(f"""
        <div class="result-card">
            <h3 style="color:#94a3b8; margin:0;">القيمة السوقية التقديرية</h3>
            <div style="font-size:3.5rem; font-weight:800; color:#2dd4bf; margin:10px 0;">{s_val}</div>
            <div style="font-size:1.2rem; color:#e2e8f0; background:rgba(255,255,255,0.1); display:inline-block; padding:5px 15px; border-radius:20px;">
                {badge} {fame_sel}
            </div>
        </div>
        """, unsafe_allow_html=True)
