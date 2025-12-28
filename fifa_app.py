
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# -------------------------------
# 1️⃣ إعداد الصفحة والتصميم
# -------------------------------
st.set_page_config(
    page_title="توقع قيمة اللاعبين | FIFA Player Value",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL and Theming
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

    * {
        font-family: 'Cairo', sans-serif;
    }
    
    .stApp {
        direction: rtl;
        text-align: right;
    }

    /* تغيير ألوان الخلفية والنصوص */
    .stAppViewContainer {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* تنسيق العناوين */
    h1, h2, h3 {
        color: #00ff88 !important; /* لون أخضر نيون */
        text-align: right;
    }

    /* الحقول والإدخالات */
    .stNumberInput, .stTextInput, .stSelectbox {
        direction: rtl;
    }
    
    div[data-baseweb="input"] {
        direction: rtl;
        border-color: #00ff88;
    }

    /* الزر الرئيسي */
    div.stButton > button {
        background-color: #00ff88;
        color: #000000;
        font-weight: bold;
        width: 100%;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: #00cc6a;
        transform: scale(1.02);
    }

    /* رسالة النجاح */
    .stSuccess {
        background-color: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        color: #00ff88;
        font-size: 20px;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }

    /* Sidebar tweaks for RTL */
    section[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 2️⃣ دوال التحميل والتدريب (Cached)
# -------------------------------
@st.cache_data
def load_data():
    path = r"C:\Users\HP\Documents\fifa deta.xlsx"
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    return df

@st.cache_resource
def train_model(df, target_column='value_eur'):
    # تنظيف البيانات الأساسي
    df_clean = df.dropna(subset=[target_column])
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # تحديد الأعمدة
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X[categorical_cols] = X[categorical_cols].astype(str)
    
    # المعالجات
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # النموذج
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X, y)
    return model, numerical_cols, categorical_cols, X

# -------------------------------
# 3️⃣ واجهة التطبيق
# -------------------------------

st.title("⚽ نظام توقع القيمة السوقية للاعبين")
st.markdown("### أدخل بيانات اللاعب للحصول على تقدير فوري للقيمة السوقية بناءً على خوارزميات الذكاء الاصطناعي")
st.markdown("---")

# تحميل البيانات
with st.spinner('جاري تحميل البيانات وتدريب النموذج... يرجى الانتظار قليلاً ⏳'):
    df = load_data()
    
if df is None:
    st.error(f"❌ لم يتم العثور على ملف البيانات في المسار: `C:\\Users\\HP\\Documents\\fifa deta.xlsx`")
    st.info("💡 يرجى التأكد من وجود الملف أو تحديث المسار.")
else:
    model, num_cols, cat_cols, X_train = train_model(df)
    
    # واجهة الإدخال
    inputs = {}
    
    # تقسيم الشاشة لعمودين
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 البيانات الرقمية (أهم 5)")
        # نأخذ أهم 5 أعمدة رقمية كما في الكود الأصلي، يمكن تعديلها لتشمل الكل
        target_num_cols = num_cols[:5] 
        for col in target_num_cols:
            default_val = float(X_train[col].median())
            inputs[col] = st.number_input(f"{col}", value=default_val)
            
    with col2:
        st.subheader("📝 البيانات الوصفية")
        # أول عمود رمزي كمثال، أو يمكن وضع المزيد
        target_cat_cols = cat_cols[:5] if cat_cols else []
        for col in target_cat_cols:
            default_val = str(X_train[col].mode()[0])
            inputs[col] = st.text_input(f"{col}", value=default_val)

    # لإكمال المدخلات التي لم تظهر في الواجهة (لضمان عمل النموذج)، نملأها بالقيم الافتراضية
    # هذا مهم جداً لأن النموذج يتوقع كل الأعمدة التي تدرب عليها
    # في الكود الأصلي للمستخدم كان يرسل only the inputs gathered, but pipeline needs all columns usually unless specified otherwise.
    # However, standard sklearn pipeline implies same input structure. 
    # Let's verify: The user code constructed a DataFrame from `inputs`. 
    # If `inputs` is missing columns that were in `X`, the model might complain or fill with NaN depending on imputer.
    # But ColumnTransformer expects the columns to be present if specified.
    # To be safe, we will fill missing columns with medians/modes from X_train.
    
    missing_cols = set(X_train.columns) - set(inputs.keys())
    for col in missing_cols:
        if col in num_cols:
            inputs[col] = X_train[col].median()
        else:
            inputs[col] = X_train[col].mode()[0]

    st.markdown("---")
    
    # وسط الشاشة للزر
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        predict_btn = st.button("🔮 توقع القيمة الآن")

    if predict_btn:
        input_df = pd.DataFrame([inputs])
        
        # التأكد من ترتيب الأعمدة كما في التدريب
        input_df = input_df[X_train.columns]
        
        with st.spinner('جاري الحساب...'):
            prediction = model.predict(input_df)[0]
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: #ffffff !important;">القيمة المتوقعة للاعب</h2>
            <div class="stSuccess">
                {prediction:,.2f} EUR 💰
            </div>
        </div>
        """, unsafe_allow_html=True)
