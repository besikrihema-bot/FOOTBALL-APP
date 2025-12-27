import streamlit as st
import random
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="تحليل أسعار لاعبي كرة القدم | Player Valuation AI",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark/Green Theme using Glassmorphism ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    
    /* General Settings */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(57, 255, 20, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(255, 215, 0, 0.05) 0%, transparent 40%);
        font-family: 'Cairo', sans-serif;
        direction: rtl;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-left: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Cairo', sans-serif !important;
        color: #f1f5f9 !important;
    }
    
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #39ff14 0%, #22c55e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stSlider {
        font-family: 'Cairo', sans-serif !important;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #39ff14 0%, #22c55e 100%);
        color: #000 !important;
        font-weight: 700 !important;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        transition: transform 0.2s, box-shadow 0.2s;
        font-size: 1.2rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(57, 255, 20, 0.4);
        border-color: #39ff14 !important;
    }
    
    /* Success/Results Box */
    .result-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(57, 255, 20, 0.3);
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .market-value {
        font-size: 3rem;
        font-weight: 700;
        color: #ffd700;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        margin: 1rem 0;
    }
    
    .player-class {
        font-size: 1.5rem;
        color: #f1f5f9;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>تحليل أسعار لاعبي كرة القدم</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; margin-top: -20px; margin-bottom: 40px;'>أداة احترافية لتقدير القيمة السوقية باستخدام المحاكاة الذكية</p>", unsafe_allow_html=True)

# --- Form Section ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 المعلومات الشخصية")
        age = st.number_input("العمر (سنة)", min_value=15, max_value=45, value=24)
        nationality = st.text_input("الجنسية", placeholder="مثال: مصري")
        foot = st.selectbox("القدم المفضلة", ["اليمنى", "اليسرى", "كلتاهما"])
        
        st.markdown("### 📏 البنية الجسدية")
        height = st.number_input("الطول (سم)", min_value=150, max_value=220, value=180)
        weight = st.number_input("الوزن (كجم)", min_value=50, max_value=120, value=75)

    with col2:
        st.markdown("### ⚽ المعلومات الكروية")
        position = st.selectbox("مركز اللعب", [
            "مهاجم صريح (ST)", "جناح (RW/LW)", "وسط هجومي (CAM)", 
            "وسط ملعب (CM)", "وسط دفاعي (CDM)", "قلب دفاع (CB)", 
            "ظهير (RB/LB)", "حارس مرمى (GK)"
        ])
        
        league_options = {
            "الدوري الإنجليزي (Premier League)": 5.0,
            "الدوري الإسباني (La Liga)": 4.8,
            "الدوري الألماني (Bundesliga)": 4.5,
            "الدوري الإيطالي (Serie A)": 4.5,
            "الدوري الفرنسي (Ligue 1)": 4.0,
            "الدوري السعودي (Roshn League)": 3.8,
            "الدوري البرتغالي/الهولندي": 3.5,
            "دوريات أخرى (أوروبا/أمريكا الجنوبية)": 3.0,
            "دوريات أضعف": 2.0
        }
        league_name = st.selectbox("الدوري الحالي", list(league_options.keys()))
        league_coef = league_options[league_name]
        
        influence_options = {
            "نجم الفريق (Key Player)": 1.2,
            "لاعب أساسي (Regular)": 1.0,
            "لاعب تدوير (Rotation)": 0.8,
            "احتياطي (Substitute)": 0.6
        }
        influence_name = st.selectbox("التأثير في الفريق", list(influence_options.keys()))
        influence_mult = influence_options[influence_name]
        
        experience = st.number_input("سنوات الخبرة", min_value=0, max_value=25, value=5)

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
    
    st.markdown("### 📊 الأداء والانضباط")
    col5, col6, col7 = st.columns(3)
    with col5:
        matches = st.number_input("مباريات آخر موسم", value=30)
    with col6:
        goals_assists = st.number_input("أهداف/صناعة", value=10)
    with col7:
        discipline = st.number_input("مستوى الانضباط (1-10)", min_value=1, max_value=10, value=8)

    # Calculate Button
    if st.button("💰 احسب القيمة السوقية"):
        with st.spinner("جاري تحليل البيانات بخوارزميات الذكاء الاصطناعي..."):
            time.sleep(1.5) # Simulate processing
            
            # --- Valuation Logic (Mirrors JS Logic) ---
            
            # 1. Weights based on position
            physWeight = 0.4
            techWeight = 0.6
            pos_key = position.split("(")[-1].replace(")", "") # ST, GK, etc.
            
            if pos_key in ['CB', 'CDM', 'GK']:
                physWeight = 0.6
                techWeight = 0.4
            
            avgPhysical = (speed + strength + stamina + agility) / 4
            avgTechnical = (skill + control + passing + shooting + vision) / 5
            overallRating = (avgPhysical * physWeight) + (avgTechnical * techWeight)
            
            # 2. Base Value (Exponential Growth)
            baseValue = 50000
            ratingFactor = pow(1.12, (overallRating - 50))
            valFromRating = baseValue * ratingFactor
            if valFromRating < baseValue:
                valFromRating = baseValue
                
            # 3. Age Factor
            ageMultiplier = 1.0
            if age < 22:
                ageMultiplier = 1.0 + ((22 - age) * 0.15)
            elif age > 28:
                ageMultiplier = max(0.1, 1.0 - ((age - 28) * 0.15))
            else:
                ageMultiplier = 1.1
            
            # 4. Performance Factor
            contributionRatio = goals_assists / matches if matches > 0 else 0
            perfMultiplier = 1.0
            
            if contributionRatio > 0.8: perfMultiplier = 1.5
            elif contributionRatio > 0.5: perfMultiplier = 1.25
            elif contributionRatio < 0.1 and pos_key in ['ST', 'RW/LW', 'CAM']: perfMultiplier = 0.8
            
            # 5. Build Penalty
            buildMultiplier = 1.0
            if pos_key == 'CB' and height < 175: buildMultiplier *= 0.8
            if pos_key == 'GK' and height < 180: buildMultiplier *= 0.7
            
            # 6. Final Calculation
            estimatedValue = valFromRating * ageMultiplier * league_coef * influence_mult * perfMultiplier * buildMultiplier
            
            # 7. Randomness
            randomVar = 0.9 + random.random() * 0.2
            estimatedValue *= randomVar
            
            if estimatedValue < 5000: estimatedValue = 5000
            
            # 8. Results
            final_value = round(estimatedValue)
            formatted_value = f"${final_value:,.0f}"
            
            # Classification
            player_class = "لاعب هاوٍ / ناشئ"
            comment = "يحتاج لتطوير كبير للوصول للمستوى الاحترافي."
            
            if final_value > 80000000:
                player_class = "أيقونة عالمية 🌍👑"
                comment = "لاعب مرشح للكرة الذهبية، قيمة تسويقية وفنية هائلة."
            elif final_value > 40000000:
                player_class = "سوبر ستار ⭐"
                comment = "نجم صف أول في أكبر الدوريات الأوروبية."
            elif final_value > 15000000:
                player_class = "لاعب دولي محترف 🔥"
                comment = "لاعب أساسي في دوريات القمة، يمتلك جودة عالية."
            elif final_value > 3000000:
                player_class = "لاعب جيد جداً ✅"
                comment = "خيار ممتاز للأندية المتوسطة في الدوريات الكبرى."
            elif final_value > 500000:
                player_class = "لاعب محترف ⚽"
                comment = "مناسب للدوريات المتوسطة أو كبديل في الفرق الكبرى."

            # League Recommendation
            league_text = "الدوريات المحلية"
            if overallRating > 85: league_text = "الدوري الإنجليزي / الإسباني (Top Tier)"
            elif overallRating > 75: league_text = "الدوري الفرنسي / الألماني / السعودي (High Tier)"
            elif overallRating > 65: league_text = "الدوري البرتغالي / التركي / البلجيكي"

            # Display
            st.markdown(f"""
            <div class="result-box">
                <h2 style="color: #39ff14;">نتائج التحليل</h2>
                <div class="market-value">{formatted_value}</div>
                <div class="player-class">{player_class}</div>
                <ul style="list-style: none; text-align: right; margin-top: 1.5rem; color: #cbd5e1;">
                    <li>📝 <strong>التقييم الفني:</strong> {int(overallRating)}/100</li>
                    <li>💡 <strong>التصنيف:</strong> {comment}</li>
                    <li>🏆 <strong>المستوى المناسب:</strong> {league_text}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
