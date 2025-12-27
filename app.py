import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="تحليل أسعار لاعبي كرة القدم | Player Valuation AI",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Preserved) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 229, 255, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(255, 215, 0, 0.05) 0%, transparent 40%);
        font-family: 'Cairo', sans-serif;
        direction: rtl;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-left: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3 { font-family: 'Cairo', sans-serif !important; color: #f1f5f9 !important; }
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #00e5ff 0%, #2979ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Sliders */
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #00e5ff !important; }
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #ffffff !important;
        border: 2px solid #00e5ff;
        width: 20px !important; height: 20px !important;
        border-radius: 50% !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }
    .stSlider label { color: #00e5ff !important; font-weight: 600; font-size: 1.1rem !important; }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00e5ff 0%, #2979ff 100%);
        color: #000 !important; font-weight: 700 !important;
        font-size: 1.2rem !important; border-radius: 0.75rem;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0, 229, 255, 0.4); }
    
    /* Results */
    .result-box {
        background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 1rem;
        padding: 2rem; text-align: center; margin-top: 2rem;
    }
    .market-value { font-size: 3rem; font-weight: 700; color: #ffd700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.3); margin: 1rem 0; }
    .player-class { font-size: 1.5rem; color: #f1f5f9; margin-bottom: 1rem; padding: 0.5rem; background: rgba(255, 255, 255, 0.05); border-radius: 0.5rem; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- ML Core Logic ---
@st.cache_resource
def train_model():
    # 1. Generate Synthetic Data
    n_samples = 2000
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(16, 40, n_samples),
        'height': np.random.normal(180, 7, n_samples),
        'league_coef': np.random.choice([2.0, 3.0, 3.5, 3.8, 4.0, 4.5, 4.8, 5.0], n_samples),
        'influence_mult': np.random.choice([0.6, 0.8, 1.0, 1.2], n_samples),
        'rating': np.random.normal(70, 10, n_samples),
        'matches': np.random.randint(0, 50, n_samples),
        'goals_assists': np.random.randint(0, 30, n_samples),
        'discipline': np.random.randint(1, 11, n_samples), # 1-10
        'injury_coef': np.random.choice([0.6, 0.7, 0.9, 1.0], n_samples),
        'fame': np.random.randint(1, 6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Target Value (Price) using a complex formula to simulate reality
    # We use this to "teach" the model the pattern
    df['performance_ratio'] = df['goals_assists'] / (df['matches'] + 1)
    
    def calculate_price(row):
        base = 50000
        rating_factor = pow(1.11, (row['rating'] - 50))
        
        # Age curve
        age_factor = 1.0
        if row['age'] < 22: age_factor = 1.0 + ((22 - row['age']) * 0.1)
        elif row['age'] > 29: age_factor = max(0.1, 1.0 - ((row['age'] - 29) * 0.15))
        
        price = (base * rating_factor * row['league_coef'] * 
                 row['influence_mult'] * row['injury_coef'] * 
                 age_factor * (1 + row['fame']*0.15))
                 
        if row['performance_ratio'] > 0.5: price *= 1.3
        
        # Add noise
        price *= np.random.uniform(0.9, 1.1)
        return int(price)

    df['market_value'] = df.apply(calculate_price, axis=1)
    
    # 2. Train Test Split
    X = df.drop(['market_value', 'performance_ratio'], axis=1) # features only
    y = df['market_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    
    return model, score

# Train model immediately
with st.spinner('جاري تدريب نموذج الذكاء الاصطناعي (Random Forest)...'):
    model, accuracy = train_model()

# --- Header ---
st.markdown("<h1>تحليل أسعار لاعبي كرة القدم</h1>", unsafe_allow_html=True)
st.caption(f"🤖 حالة النموذج: مدرب وجاهز | 📊 دقة الاختبار (R²): {accuracy:.2f}")

# --- Form ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 المعلومات الشخصية")
        age = st.number_input("العمر", 15, 45, 24)
        nationality = st.text_input("الجنسية", placeholder="مثال: مصري")
        foot = st.selectbox("القدم المفضلة", ["اليمنى", "اليسرى", "كلتاهما"])
        st.markdown("### 📏 البنية الجسدية")
        height = st.number_input("الطول (سم)", 150, 220, 180)
        weight = st.number_input("الوزن (كجم)", 50, 120, 75)

    with col2:
        st.markdown("### ⚽ المعلومات الكروية")
        position = st.selectbox("مركز اللعب", [
            "مهاجم صريح (ST)", "جناح (RW/LW)", "وسط هجومي (CAM)", 
            "وسط ملعب (CM)", "وسط دفاعي (CDM)", "قلب دفاع (CB)", 
            "ظهير (RB/LB)", "حارس مرمى (GK)"
        ])
        
        league_options = {
            "الدوري الإنجليزي (Premier League)": 5.0, "الدوري الإسباني (La Liga)": 4.8,
            "الدوري الألماني (Bundesliga)": 4.5, "الدوري الإيطالي (Serie A)": 4.5,
            "الدوري الفرنسي (Ligue 1)": 4.0, "الدوري السعودي (Roshn League)": 3.8,
            "الدوري البرتغالي/الهولندي": 3.5, "دوريات أخرى": 3.0, "دوريات أضعف": 2.0
        }
        league_name = st.selectbox("الدوري الحالي", list(league_options.keys()))
        league_coef = league_options[league_name]
        
        influence_options = {
            "نجم الفريق (Key Player)": 1.2, "لاعب أساسي (Regular)": 1.0,
            "لاعب تدوير (Rotation)": 0.8, "احتياطي (Substitute)": 0.6
        }
        influence_name = st.selectbox("التأثير في الفريق", list(influence_options.keys()))
        influence_mult = influence_options[influence_name]
        
        experience = st.number_input("سنوات الخبرة", 0, 25, 5)

    st.divider()
    
    # Attributes
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### ⚡ القدرات البدنية")
        speed = st.slider("السرعة", 0, 100, 70)
        strength = st.slider("القوة البدنية", 0, 100, 70)
        stamina = st.slider("التحمل", 0, 100, 70)
        agility = st.slider("الرشاقة", 0, 100, 70)
    with col4:
        st.markdown("### 🎯 القدرات الفنية")
        skill = st.slider("المهارة", 0, 100, 70)
        control = st.slider("التحكم بالكرة", 0, 100, 70)
        passing = st.slider("التمرير", 0, 100, 70)
        shooting = st.slider("التسديد", 0, 100, 70)
        vision = st.slider("الرؤية", 0, 100, 70)

    st.divider()
    
    st.markdown("### 📢 الشهرة والأداء")
    col5, col6, col7 = st.columns(3)
    with col5: matches = st.number_input("مباريات آخر موسم", value=30)
    with col6: goals_assists = st.number_input("أهداف/صناعة", value=10)
    with col7: fame = st.slider("الشهرة الجماهيرية", 1, 5, 2)

    st.markdown("### 🏥 حالة طبية و انضباط")
    col8, col9 = st.columns(2)
    with col8: discipline = st.number_input("الانضباط (1-10)", 1, 10, 8)
    with col9:
        injury_status = st.selectbox("تاريخ الإصابات", ["سليم تماماً", "إصابات طفيفة", "متكرر الإصابات", "عائد من رباط صليبي"])
        injury_map = {"سليم تماماً": 1.0, "إصابات طفيفة": 0.9, "متكرر الإصابات": 0.7, "عائد من رباط صليبي": 0.6}
        injury_coef = injury_map[injury_status]

    if st.button("💰 احسب القيمة السوقية (AI)"):
        with st.spinner("جاري التنبؤ بالقيمة باستخدام النموذج..."):
            time.sleep(1)
            
            # Prepare Input Vector for Model
            # Needs to calculate 'rating' first as feature
            physWeight = 0.6 if position in ["قلب دفاع (CB)", "وسط دفاعي (CDM)", "حارس مرمى (GK)"] else 0.4
            techWeight = 1 - physWeight
            avgPhysical = (speed + strength + stamina + agility) / 4
            avgTechnical = (skill + control + passing + shooting + vision) / 5
            overallRating = (avgPhysical * physWeight) + (avgTechnical * techWeight)
            
            # [age, height, league_coef, influence_mult, rating, matches, goals_assists, discipline, injury_coef, fame]
            input_features = pd.DataFrame([{
                'age': age,
                'height': height,
                'league_coef': league_coef,
                'influence_mult': influence_mult,
                'rating': overallRating,
                'matches': matches,
                'goals_assists': goals_assists,
                'discipline': discipline,
                'injury_coef': injury_coef,
                'fame': fame
            }])
            
            # Predict
            prediction = model.predict(input_features)[0]
            
            # Post-process (Build penalties logic tailored for specific positions can be applied on top if model data didn't catch it fully, 
            # but ideally model catches it. For now leaving pure prediction is better for ML authenticity)
            
            final_value = round(prediction)
            formatted_value = f"${final_value:,.0f}"
            
            # Class Logic
            player_class = "لاعب هاوٍ / ناشئ"
            comment = "يحتاج لتطوير كبير."
            if final_value > 80000000: player_class, comment = "أيقونة عالمية 🌍👑", "مرشح للكرة الذهبية."
            elif final_value > 40000000: player_class, comment = "سوبر ستار ⭐", "نجم صف أول."
            elif final_value > 15000000: player_class, comment = "لاعب دولي محترف 🔥", "جودة عالية."
            elif final_value > 3000000: player_class, comment = "لاعب جيد جداً ✅", "خيار ممتاز."
            elif final_value > 500000: player_class, comment = "لاعب محترف ⚽", "جيد للدوريات المتوسطة."

            st.markdown(f"""
            <div class="result-box">
                <h2 style="color: #00e5ff;">نتائج التنبؤ (Random Forest)</h2>
                <div class="market-value">{formatted_value}</div>
                <div class="player-class">{player_class}</div>
                <ul style="list-style: none; text-align: right; margin-top: 1.5rem; color: #cbd5e1;">
                    <li>📝 <strong>التقييم الفني:</strong> {int(overallRating)}/100</li>
                    <li>💡 <strong>التصنيف:</strong> {comment}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
