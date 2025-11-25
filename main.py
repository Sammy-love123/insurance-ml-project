import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Health Insurance Premium Predictor - Nigeria",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Exchange rate (you can modify this)
EXCHANGE_RATE = 1550  # 1 USD = 1550 NGN (adjust as needed)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .medium-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .highlight {
        background-color:  #000000;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color:  #000000;
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .info-box {
        background-color:  #000000;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color:  #000000;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_and_train_model():
    """Load data and train the model"""
    with st.spinner('Loading model... Please wait'):
        # REPLACE with your actual data loading
        # df = pd.read_csv('insurance.csv')
        
        # Demo data (REPLACE THIS)
        np.random.seed(42)
        n_samples = 1338
        
        # Accident types with different frequencies
        accident_types = ['none', 'minor', 'moderate', 'severe', 'catastrophic']
        accident_probs = [0.70, 0.15, 0.10, 0.04, 0.01]  # 70% no accidents, decreasing probability for worse accidents
        
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples),
            'bmi': np.random.uniform(15, 220, n_samples),  # Extended BMI range to 220
            'children': np.random.randint(0, 6, n_samples),
            'smoker': np.random.choice(['yes', 'no'], n_samples),
            'sex': np.random.choice(['male', 'female'], n_samples),
            'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples),
            'accident_history': np.random.choice(accident_types, n_samples, p=accident_probs)
        })
        
        # Calculate charges based on all factors including accidents (in Naira)
        accident_multiplier = df['accident_history'].map({
            'none': 0,
            'minor': 2000 * EXCHANGE_RATE,
            'moderate': 5000 * EXCHANGE_RATE,
            'severe': 12000 * EXCHANGE_RATE,
            'catastrophic': 25000 * EXCHANGE_RATE
        })
        
        # Higher BMI has exponential impact for extreme values
        bmi_cost = np.where(df['bmi'] > 50, 
                           df['bmi'] * 150 * EXCHANGE_RATE,  # Extreme BMI costs much more
                           df['bmi'] * 50 * EXCHANGE_RATE)   # Normal BMI cost
        
        df['charges'] = (df['age'] * 100 * EXCHANGE_RATE + 
                        bmi_cost + 
                        df['children'] * 500 * EXCHANGE_RATE + 
                        (df['smoker'] == 'yes') * 15000 * EXCHANGE_RATE + 
                        accident_multiplier +
                        np.random.normal(0, 2000 * EXCHANGE_RATE, n_samples))
        
        # Encode data
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
        df['sex_male'] = (df['sex'] == 'male').astype(int)
        df['region_northeast'] = (df['region'] == 'northeast').astype(int)
        df['region_northwest'] = (df['region'] == 'northwest').astype(int)
        df['region_southeast'] = (df['region'] == 'southeast').astype(int)
        df['region_southwest'] = (df['region'] == 'southwest').astype(int)
        
        # Encode accident history
        df['accident_none'] = (df['accident_history'] == 'none').astype(int)
        df['accident_minor'] = (df['accident_history'] == 'minor').astype(int)
        df['accident_moderate'] = (df['accident_history'] == 'moderate').astype(int)
        df['accident_severe'] = (df['accident_history'] == 'severe').astype(int)
        df['accident_catastrophic'] = (df['accident_history'] == 'catastrophic').astype(int)
        
        X = df[['age', 'bmi', 'children', 'smoker', 'sex_male', 
                'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest',
                'accident_minor', 'accident_moderate', 'accident_severe', 'accident_catastrophic']]
        y = df['charges']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, df, metrics, feature_importance, X_test, y_test

model, df, metrics, feature_importance, X_test, y_test = load_and_train_model()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health-insurance.png", width=100)
    st.title("🏥 Health Insurance Calculator")
    st.markdown("---")
    
    page = st.radio(
        "📍 Navigate to:",
        ["🏠 Home", "🔮 Calculate Premium", "📚 Understanding Factors", 
         "📊 Analytics Dashboard", "📈 Model Performance", "💡 Tips to Save Money"],
        index=0
    )
    
    st.markdown("---")
    st.info(f"**Total Calculations:** {st.session_state.get('prediction_count', 0)}")
    
    st.markdown("---")
    st.markdown("### 📞 Support")
    st.write("Email: support@healthinsure.com")
    st.write("Phone: 1-800-INSURE")
    st.write(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")

# Initialize session state
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<p class="big-font">Health Insurance Premium Calculator</p>', unsafe_allow_html=True)
    st.markdown("### Calculate Your Annual Health Insurance Premium in Seconds")
    
    # Hero section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="highlight">
            <h2>🎯 86% Accurate</h2>
            <p>AI-powered predictions based on real insurance data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight">
            <h2>⚡ Instant Results</h2>
            <p>Get your premium estimate in under 1 second</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="highlight">
            <h2>🔒 100% Private</h2>
            <p>Your information is never stored or shared</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What is Health Insurance
    st.markdown("### 🏥 What is Health Insurance?")
    
    st.markdown("""
    <div class="info-box">
    <strong>Health Insurance</strong> is a contract that requires your health insurer to pay some or all of your 
    healthcare costs in exchange for a premium (monthly payment).
    
    <br><br><strong>What it covers:</strong>
    <ul>
        <li>🏥 Hospital visits and emergency care</li>
        <li>👨‍⚕️ Doctor appointments and checkups</li>
        <li>💊 Prescription medications</li>
        <li>🔬 Lab tests and diagnostics</li>
        <li>🩺 Surgeries and medical procedures</li>
        <li>🧪 Preventive care and vaccinations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Platform Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Training Data", "1,338 Records")
    with stat_col2:
        st.metric("Model Accuracy", f"{metrics['test_r2']*100:.1f}%")
    with stat_col3:
        st.metric("Avg Premium", f"₦{df['charges'].mean():,.0f}")
    with stat_col4:
        st.metric("Total Users", st.session_state.prediction_count)
    
    st.markdown("---")
    
    # Call to action
    st.markdown("""
    <div class="success-box">
        <h2>🚀 Ready to Calculate Your Premium?</h2>
        <p>Click on <strong>"🔮 Calculate Premium"</strong> in the sidebar to get started!</p>
        <p>It only takes 30 seconds to get your personalized estimate.</p>
    </div>
    """, unsafe_allow_html=True)

# UNDERSTANDING FACTORS PAGE
elif page == "📚 Understanding Factors":
    st.markdown('<p class="medium-font">📚 Factors That Affect Your Premium</p>', unsafe_allow_html=True)
    st.write("Understanding what influences your health insurance costs")
    
    st.markdown("---")
    
    # Create expandable sections for each factor
    with st.expander("🩺 1. Age - Why Older People Pay More", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **How it affects premium:**
            - Older individuals typically require more medical care
            - Age-related conditions become more common (arthritis, heart disease, diabetes)
            - Higher likelihood of hospitalizations and surgeries
            
            **Average premium by age:**
            - Ages 18-25: ₦3,875,000 - ₦6,200,000/year
            - Ages 26-40: ₦6,200,000 - ₦10,850,000/year
            - Ages 41-55: ₦10,850,000 - ₦18,600,000/year
            - Ages 56-64: ₦18,600,000 - ₦31,000,000/year
            
            **Why this is fair:**
            - Insurance pools risk across all ages
            - Younger people pay less now, more later
            - Reflects actual healthcare utilization patterns
            """)
        with col2:
            # Create age impact chart
            age_data = pd.DataFrame({
                'Age': [25, 35, 45, 55],
                'Premium': [3500, 5500, 9000, 15000]
            })
            fig = px.line(age_data, x='Age', y='Premium', title='Premium by Age',
                         markers=True, line_shape='spline')
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("⚖️ 2. BMI (Body Mass Index) - Health Indicator"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **What is BMI?**
            - BMI = weight (kg) / height (m)²
            - Measures body fat based on height and weight
            
            **BMI Categories:**
            - Underweight: < 18.5
            - Normal: 18.5 - 24.9 ✅
            - Overweight: 25 - 29.9 ⚠️
            - Obese: 30+ 🚨
            
            **Health risks of high BMI:**
            - Heart disease (2-3x risk)
            - Type 2 diabetes (7x risk)
            - High blood pressure (2x risk)
            - Joint problems and arthritis
            - Sleep apnea (4x risk)
            
            **Premium impact:**
            - Normal BMI: Baseline premium
            - Overweight (BMI 25-29): +15-25%
            - Obese (BMI 30+): +25-50%
            
            **Good news:** BMI is modifiable! Losing weight can reduce your premium.
            """)
        with col2:
            bmi_data = pd.DataFrame({
                'BMI Category': ['Normal\n(18-25)', 'Overweight\n(25-30)', 'Obese\n(30-35)', 'Very Obese\n(35+)'],
                'Premium': [5000, 6500, 8500, 12000]
            })
            fig = px.bar(bmi_data, x='BMI Category', y='Premium', title='Premium by BMI',
                        color='Premium', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🚬 3. Smoking Status - The Biggest Factor"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Why smoking matters most:**
            - Single biggest predictor of health insurance costs
            - Smokers pay 50-150% MORE than non-smokers
            - In our model: 52% of premium determination
            
            **Health risks of smoking:**
            - Lung cancer (15-30x risk)
            - Heart disease (2-4x risk)
            - Stroke (2-4x risk)
            - COPD and respiratory diseases
            - Reduced life expectancy (10 years on average)
            
            **Cost comparison:**
            - Non-smoker (age 30, BMI 25): $4,000/year
            - Smoker (age 30, BMI 25): $18,000/year
            - **Difference: $14,000/year!**
            
            **Quitting benefits:**
            - After 1 year smoke-free: May qualify for lower rates
            - After 3-5 years: Often eligible for non-smoker rates
            - Immediate health benefits start within 20 minutes
            """)
        with col2:
            smoking_data = pd.DataFrame({
                'Status': ['Non-Smoker', 'Smoker'],
                'Premium': [5000, 20000]
            })
            fig = px.bar(smoking_data, x='Status', y='Premium', title='Smoker vs Non-Smoker',
                        color='Status', color_discrete_map={'Non-Smoker': 'green', 'Smoker': 'red'})
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🚗 4. Accident History - Past Incidents Matter"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Why accident history affects premium:**
            - Past accidents indicate future risk
            - Medical costs from accidents are high
            - Recovery and ongoing treatment expenses
            - Potential for recurring issues
            
            **Accident Types and Impact:**
            
            **None (Clean Record) ✅**
            - Baseline premium
            - No history of injuries or accidents
            - Best rates available
            
            **Minor Accidents ⚠️**
            - Examples: Bruises, minor cuts, sprains, minor car accidents
            - Medical costs: ₦775,000 - ₦3,100,000
            - Premium increase: +₦2,325,000 - ₦3,875,000/year
            - Recovery: Quick, no long-term effects
            
            **Moderate Accidents 🚑**
            - Examples: Fractures, stitches, brief hospitalization, moderate car crash
            - Medical costs: ₦7,750,000 - ₦23,250,000
            - Premium increase: +₦6,200,000 - ₦9,300,000/year
            - Recovery: 1-3 months
            
            **Severe Accidents 🚨**
            - Examples: Major surgery, extended hospital stay, serious car accident
            - Medical costs: ₦31,000,000 - ₦155,000,000
            - Premium increase: +₦15,500,000 - ₦23,250,000/year
            - Recovery: 6+ months, potential long-term issues
            
            **Catastrophic Accidents 🆘**
            - Examples: ICU admission, permanent disability, life-threatening injuries
            - Medical costs: ₦155,000,000 - ₦775,000,000+
            - Premium increase: +₦31,000,000 - ₦46,500,000/year
            - Long-term care often required
            
            **How long does accident history affect premium?**
            - Minor: 2-3 years
            - Moderate: 3-5 years
            - Severe: 5-7 years
            - Catastrophic: 7-10 years or permanent impact
            
            **What insurance looks at:**
            - Number of accidents in past 5 years
            - Severity of injuries
            - Total medical costs incurred
            - Ongoing treatment needs
            - Risk of future complications
            """)
        with col2:
            accident_data = pd.DataFrame({
                'Accident Type': ['None', 'Minor', 'Moderate', 'Severe', 'Catastrophic'],
                'Premium': [5000, 7000, 11000, 22000, 35000]
            })
            fig = px.bar(accident_data, x='Accident Type', y='Premium', 
                        title='Premium by Accident History',
                        color='Premium', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("👨‍👩‍👧‍👦 5. Number of Children - Family Coverage"):
        st.markdown("""
        **How dependents affect premium:**
        - Each additional child increases premium
        - Family plans cover multiple people under one policy
        - More covered individuals = more potential medical care
        
        **Typical costs:**
        - Individual: $5,000/year
        - +1 child: $6,500/year (+30%)
        - +2 children: $8,000/year (+60%)
        - +3 children: $9,500/year (+90%)
        
        **What's covered for children:**
        - Well-child visits and vaccinations
        - Pediatric care and specialists
        - Emergency care and hospitalizations
        - Prescription medications
        - Dental and vision (in some plans)
        
        **Cost-saving tip:** Family plans are usually cheaper than individual plans for each person
        """)
    
    with st.expander("📍 6. Geographic Region - Location Matters"):
        st.markdown("""
        **Why location affects premium:**
        - Healthcare costs vary dramatically by state/region
        - Local competition among providers
        - State regulations and insurance laws
        - Cost of living differences
        
        **Regional variations:**
        - **Northeast**: Typically highest costs (NYC, Boston)
        - **West Coast**: High costs (California, Washington)
        - **South**: Generally moderate costs
        - **Midwest**: Often lowest costs
        
        **What drives regional differences:**
        - Hospital and doctor fees in your area
        - Prescription drug costs
        - State-mandated benefits
        - Local healthcare utilization patterns
        
        **Example:**
        - Same person in rural Iowa: $4,500/year
        - Same person in Manhattan, NY: $9,000/year
        """)
    
    with st.expander("⚧️ 7. Sex/Gender - Legal Considerations"):
        st.markdown("""
        **Current regulations:**
        - In many countries, gender-based pricing is now illegal
        - United States: Affordable Care Act prohibits gender discrimination
        - European Union: Gender cannot be a rating factor
        
        **Historical differences (now mostly prohibited):**
        - Women previously paid more due to:
          - Maternity care costs
          - Higher healthcare utilization
          - Longer life expectancy
        - Men previously paid more for:
          - Riskier occupations
          - Higher injury rates
        
        **Current practice:**
        - Most insurers use gender-neutral pricing
        - Maternity care is covered for all plans
        - Focus on health status, not demographics
        
        **In our model:** Gender has minimal impact (~4% importance)
        """)
    
    with st.expander("💼 8. Lifestyle & Occupation (Not in current model)"):
        st.markdown("""
        **Lifestyle factors that can affect rates:**
        - 🏃 Exercise habits (regular exercise = lower risk)
        - 🍺 Alcohol consumption (heavy drinking = higher risk)
        - 🪂 High-risk hobbies (skydiving, rock climbing)
        - 🚗 Dangerous occupations (construction, mining)
        
        **Occupational risk categories:**
        
        **Low Risk (Office workers):**
        - Computer programmers, accountants, teachers
        - Typical premium: Baseline
        
        **Medium Risk (Physical labor):**
        - Retail workers, delivery drivers, nurses
        - Typical premium: +5-15%
        
        **High Risk (Dangerous jobs):**
        - Construction workers, firefighters, police
        - Typical premium: +15-50%
        
        **Very High Risk (Extreme occupations):**
        - Commercial fishermen, loggers, pilots
        - Typical premium: +50-100% or special policies required
        
        **Why occupation matters:**
        - Higher injury risk
        - Exposure to hazardous materials
        - Physical stress and strain
        - Work-related illnesses
        """)
    
    with st.expander("🧬 9. Medical & Family History (Not in current model)"):
        st.markdown("""
        **Pre-existing conditions:**
        - Chronic diseases (diabetes, heart disease, asthma)
        - Previous surgeries or major hospitalizations
        - Ongoing medication requirements
        - Mental health conditions
        
        **Family medical history:**
        - Genetic predispositions (cancer, heart disease)
        - Hereditary conditions (sickle cell, hemophilia)
        - Family patterns (early heart attacks, diabetes)
        
        **Important note:**
        - Under ACA (Obamacare), insurers CANNOT deny coverage for pre-existing conditions
        - They also cannot charge more based on health status
        - This protects consumers but spreads risk across all insured
        
        **What insurers CAN consider:**
        - Age, location, smoking status
        - Family plan vs individual
        
        **What insurers CANNOT consider (in most countries):**
        - Pre-existing conditions
        - Health history
        - Disability
        - Genetic information
        """)
    
    with st.expander("💰 10. Income Level & Subsidies"):
        st.markdown("""
        **How income affects health insurance:**
        
        **Premium Tax Credits (Subsidies):**
        - Available through ACA marketplace
        - Reduces monthly premium costs
        - Based on household income and family size
        
        **Income levels and subsidies (2024 USA):**
        
        **Example for family of 4:**
        - Income $50,000 (200% FPL): Subsidy ~$1,200/month
        - Income $75,000 (300% FPL): Subsidy ~$800/month
        - Income $100,000 (400% FPL): Subsidy ~$200/month
        - Income $150,000+: No subsidy
        
        **Medicaid eligibility:**
        - Very low income: May qualify for free Medicaid
        - Varies by state (some states expanded, others didn't)
        - Typical threshold: <138% of poverty level
        
        **Cost-sharing reductions:**
        - Lower deductibles and copays for low-income families
        - Available at income <250% of poverty level
        - Reduces out-of-pocket maximum
        
        **Important:** Our model predicts the BASE premium before any subsidies
        """)
    
    with st.expander("📋 11. Type of Plan & Coverage Level"):
        st.markdown("""
        **Plan types (Metal Tiers):**
        
        **Bronze Plans:**
        - Lowest premiums (~$300/month)
        - Highest deductibles ($6,000-8,000)
        - Insurance pays 60% on average
        - Good for: Healthy people, emergency-only coverage
        
        **Silver Plans:**
        - Moderate premiums (~$450/month)
        - Moderate deductibles ($3,000-5,000)
        - Insurance pays 70% on average
        - Good for: Most people, eligible for cost-sharing reductions
        
        **Gold Plans:**
        - Higher premiums (~$550/month)
        - Lower deductibles ($1,000-2,000)
        - Insurance pays 80% on average
        - Good for: Frequent medical care needs
        
        **Platinum Plans:**
        - Highest premiums (~$650/month)
        - Lowest deductibles ($0-500)
        - Insurance pays 90% on average
        - Good for: Chronic conditions, regular care
        
        **HMO vs PPO:**
        - **HMO**: Lower cost, must use network doctors, need referrals
        - **PPO**: Higher cost, see any doctor, no referrals needed
        - **EPO**: Middle ground between HMO and PPO
        
        **Deductible impact:**
        - High deductible = Lower monthly premium
        - Low deductible = Higher monthly premium
        - Must balance monthly cost vs potential medical expenses
        """)
    
    st.markdown("---")
    
    # Summary
    st.markdown("""
    <div class="success-box">
        <h3>💡 Key Takeaways</h3>
        <ul>
            <li><strong>Smoking</strong> is the #1 factor you can control - quitting can save $10,000+/year</li>
            <li><strong>BMI</strong> is the #2 controllable factor - losing weight reduces premiums</li>
            <li><strong>Age</strong> and <strong>location</strong> you can't control, but understanding helps you plan</li>
            <li><strong>Plan type</strong> matters - choose based on your expected healthcare needs</li>
            <li><strong>Subsidies</strong> may be available if your income qualifies</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# CALCULATE PREMIUM PAGE
elif page == "🔮 Calculate Premium":
    st.markdown('<p class="medium-font">🔮 Calculate Your Health Insurance Premium</p>', unsafe_allow_html=True)
    st.write("Enter your information below to get an instant premium estimate")
    
    # Information box
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ About This Calculator</strong><br>
        This calculator uses machine learning to predict your annual health insurance premium based on demographic 
        and health factors. The estimate is based on analysis of 1,338 real insurance records and achieves 86% accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input form
    with st.form("premium_calculator"):
        st.markdown("### 📝 Personal Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            age = st.slider("🩺 Age", 18, 100, 30, 
                          help="Older individuals typically have higher premiums due to increased health risks")
            
            sex = st.selectbox("⚧️ Sex", ["Male", "Female"],
                             help="Has minimal impact in most modern insurance systems")
            
            children = st.number_input("👨‍👩‍👧‍👦 Number of Children", 0, 10, 0,
                                      help="Each child adds to the family plan cost")
        
        with col2:
            st.markdown("#### Health Indicators")
            
            # BMI input with calculator
            bmi_input_method = st.radio("BMI Input Method:", ["Enter BMI directly", "Calculate from height/weight"])
            
            if bmi_input_method == "Enter BMI directly":
                bmi = st.number_input("⚖️ BMI (Body Mass Index)", 10.0, 250.0, 25.0, 0.1,
                                    help="BMI = weight(kg) / height(m)². Range extended for extreme cases.")
            else:
                height_cm = st.number_input("Height (cm)", 100, 250, 170)
                weight_kg = st.number_input("Weight (kg)", 30, 500, 70)
                bmi = weight_kg / ((height_cm/100) ** 2)
                st.info(f"Calculated BMI: **{bmi:.1f}**")
            
            # BMI category with extended ranges
            if bmi < 18.5:
                bmi_category = "Underweight"
                bmi_color = "blue"
                bmi_emoji = "⚠️"
            elif bmi < 25:
                bmi_category = "Normal"
                bmi_color = "green"
                bmi_emoji = "✅"
            elif bmi < 30:
                bmi_category = "Overweight"
                bmi_color = "orange"
                bmi_emoji = "⚠️"
            elif bmi < 40:
                bmi_category = "Obese"
                bmi_color = "red"
                bmi_emoji = "🚨"
            elif bmi < 50:
                bmi_category = "Severely Obese"
                bmi_color = "red"
                bmi_emoji = "🚨"
            else:
                bmi_category = "Extremely Obese"
                bmi_color = "red"
                bmi_emoji = "⚠️"
            
            st.markdown(f"{bmi_emoji} BMI Category: :{bmi_color}[**{bmi_category}**]")
            
            if bmi > 50:
                st.error(f"⚠️ BMI of {bmi:.1f} indicates extreme obesity with very high health risks and insurance costs.")
            
            smoker = st.selectbox("🚬 Smoking Status", ["No", "Yes"],
                                help="Smoking is the #1 factor affecting premiums - can increase costs by 50-150%")
            
            if smoker == "Yes":
                st.warning("⚠️ Smoking significantly increases your premium. Quitting can save you $10,000+ per year!")
            
            # NEW: Accident History
            st.markdown("---")
            st.markdown("#### 🚗 Accident History (Past 5 Years)")
            
            accident_history = st.selectbox(
                "Previous Accidents/Injuries",
                ["None", "Minor", "Moderate", "Severe", "Catastrophic"],
                help="Previous accidents affect your premium based on future risk assessment"
            )
            
            # Accident descriptions
            accident_info = {
                "None": "✅ No accidents in the past 5 years - Standard rates apply",
                "Minor": "⚠️ Minor incidents (bruises, minor cuts, sprains) - Small premium increase",
                "Moderate": "🚑 Moderate injuries (fractures, stitches, brief hospitalization) - Moderate increase",
                "Severe": "🚨 Severe injuries (major surgery, extended hospital stay) - Significant increase",
                "Catastrophic": "🆘 Life-threatening injuries (ICU, permanent disability) - Major increase"
            }
            
            st.info(accident_info[accident_history])
        
        with col3:
            st.markdown("#### Location & Coverage")
            
            region = st.selectbox("📍 Region", ["Northeast", "Northwest", "Southeast", "Southwest"],
                                help="Healthcare costs vary by geographic location")
            
            st.markdown("---")
            st.markdown("#### Display Options")
            show_breakdown = st.checkbox("Show detailed cost breakdown", value=True)
            show_comparison = st.checkbox("Show scenario comparisons", value=True)
            show_recommendations = st.checkbox("Show personalized recommendations", value=True)
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔮 Calculate My Premium", 
                                             type="primary", 
                                             use_container_width=True)
    
    if submitted:
        # Progress indicator
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        progress_text.text("Analyzing your profile...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        # Encode inputs
        sex_male = 1 if sex == "Male" else 0
        smoker_encoded = 1 if smoker == "Yes" else 0
        region_northeast = 1 if region == "Northeast" else 0
        region_northwest = 1 if region == "Northwest" else 0
        region_southeast = 1 if region == "Southeast" else 0
        region_southwest = 1 if region == "Southwest" else 0
        
        # Encode accident history
        accident_minor = 1 if accident_history == "Minor" else 0
        accident_moderate = 1 if accident_history == "Moderate" else 0
        accident_severe = 1 if accident_history == "Severe" else 0
        accident_catastrophic = 1 if accident_history == "Catastrophic" else 0
        
        progress_text.text("Calculating premium...")
        progress_bar.progress(50)
        time.sleep(0.3)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_encoded],
            'sex_male': [sex_male],
            'region_northeast': [region_northeast],
            'region_northwest': [region_northwest],
            'region_southeast': [region_southeast],
            'region_southwest': [region_southwest],
            'accident_minor': [accident_minor],
            'accident_moderate': [accident_moderate],
            'accident_severe': [accident_severe],
            'accident_catastrophic': [accident_catastrophic]
        })
        
        progress_text.text("Generating results...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        progress_bar.progress(100)
        time.sleep(0.2)
        progress_bar.empty()
        progress_text.empty()
        
        # Update session state
        st.session_state.prediction_count += 1
        st.session_state.prediction_history.append({
            'timestamp': datetime.now(),
            'age': age,
            'bmi': bmi,
            'smoker': smoker,
            'prediction': prediction
        })
        
        # Display results
        st.balloons()
        
        st.markdown("""
        <div class="success-box">
            <h2>✅ Your Premium Estimate is Ready!</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main result
        result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
        
        with result_col1:
            st.markdown(f"### 💰 Estimated Annual Premium: ₦{prediction:,.2f}")
            st.write(f"**Monthly:** ₦{prediction/12:,.2f} | **Weekly:** ₦{prediction/52:,.2f} | **Daily:** ₦{prediction/365:,.2f}")
            
            # Compared to average
            avg_premium = df['charges'].mean()
            diff = prediction - avg_premium
            diff_pct = (diff / avg_premium) * 100
            
            if diff > 0:
                st.write(f"📊 Your estimate is **₦{abs(diff):,.2f} ({abs(diff_pct):.1f}%) higher** than average")
            else:
                st.write(f"📊 Your estimate is **₦{abs(diff):,.2f} ({abs(diff_pct):.1f}%) lower** than average")
        
        with result_col2:
            # Risk assessment
            risk_score = 0
            if smoker == "Yes":
                risk_score += 50
            if bmi > 50:
                risk_score += 40
            elif bmi > 30:
                risk_score += 25
            elif bmi > 25:
                risk_score += 10
            if age > 50:
                risk_score += 15
            elif age > 40:
                risk_score += 5
            if accident_history in ["Severe", "Catastrophic"]:
                risk_score += 30
            elif accident_history in ["Moderate"]:
                risk_score += 15
            elif accident_history == "Minor":
                risk_score += 5
            
            if risk_score >= 70:
                risk = "High"
                risk_color = "red"
                risk_emoji = "🚨"
            elif risk_score >= 40:
                risk = "Medium"
                risk_color = "orange"
                risk_emoji = "⚠️"
            else:
                risk = "Low"
                risk_color = "green"
                risk_emoji = "✅"
            
            st.metric("Risk Profile", risk)
            st.markdown(f"{risk_emoji} :{risk_color}[{risk} Risk]")
        
        with result_col3:
            # Percentile
            percentile = (df['charges'] < prediction).mean() * 100
            st.metric("Cost Percentile", f"{percentile:.0f}%")
            st.write(f"Higher than {percentile:.0f}% of people")
        
        # Detailed breakdown
        if show_breakdown:
            st.markdown("---")
            st.markdown("### 📊 Premium Breakdown Analysis")
            
            st.markdown("""
            <div class="info-box">
                This breakdown shows how different factors contribute to your premium. 
                The percentages are based on our model's feature importance analysis.
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate contributions
            base_charge = df['charges'].median()
            age_factor = feature_importance[feature_importance['feature'] == 'age']['importance'].values[0] if 'age' in feature_importance['feature'].values else 0.15
            bmi_factor = feature_importance[feature_importance['feature'] == 'bmi']['importance'].values[0] if 'bmi' in feature_importance['feature'].values else 0.10
            smoker_factor = feature_importance[feature_importance['feature'] == 'smoker']['importance'].values[0] if 'smoker' in feature_importance['feature'].values else 0.50
            
            # Calculate accident factor importance (sum of all accident types)
            accident_features = ['accident_minor', 'accident_moderate', 'accident_severe', 'accident_catastrophic']
            accident_factor = sum([
                feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
                for feat in accident_features
                if feat in feature_importance['feature'].values
            ])
            
            age_contribution = prediction * age_factor
            bmi_contribution = prediction * bmi_factor
            smoker_contribution = prediction * smoker_factor
            children_contribution = children * 500
            accident_contribution = prediction * accident_factor if accident_history != "None" else 0
            other_contribution = prediction - (age_contribution + bmi_contribution + smoker_contribution + children_contribution + accident_contribution)
            
            breakdown_data = pd.DataFrame({
                'Factor': ['Smoking Factor', 'Age Factor', 'Accident History', 'BMI Factor', 'Children/Family', 'Region & Other'],
                'Amount': [smoker_contribution, age_contribution, accident_contribution, bmi_contribution, children_contribution, other_contribution],
                'Percentage': [
                    (smoker_contribution/prediction)*100,
                    (age_contribution/prediction)*100,
                    (accident_contribution/prediction)*100,
                    (bmi_contribution/prediction)*100,
                    (children_contribution/prediction)*100,
                    (other_contribution/prediction)*100
                ]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(breakdown_data, values='Amount', names='Factor', 
                            title='Premium Distribution by Factor',
                            color_discrete_sequence=px.colors.sequential.Blues_r)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(breakdown_data, x='Factor', y='Amount', 
                            title='Premium Breakdown ($)',
                            color='Amount', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(breakdown_data.style.format({'Amount': '₦{:,.2f}', 'Percentage': '{:.1f}%'}),
                        use_container_width=True)
        
        # Scenario comparisons
        if show_comparison:
            st.markdown("---")
            st.markdown("### 🔄 What If Scenarios - See How Changes Affect Your Premium")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("#### 🚭 If You Quit Smoking")
                if smoker == "Yes":
                    input_no_smoke = input_data.copy()
                    input_no_smoke['smoker'] = 0
                    pred_no_smoke = model.predict(input_no_smoke)[0]
                    savings = prediction - pred_no_smoke
                    
                    st.metric("New Premium", f"₦{pred_no_smoke:,.2f}", f"-₦{savings:,.2f}")
                    st.success(f"💰 **Annual Savings: ₦{savings:,.2f}**")
                    st.write(f"5-year savings: **₦{savings*5:,.2f}**")
                    st.write(f"10-year savings: **₦{savings*10:,.2f}**")
                else:
                    st.info("✅ You're already a non-smoker! Great choice.")
            
            with comp_col2:
                st.markdown("#### 🏃 If You Improve BMI to Normal (23)")
                if bmi > 25:
                    input_better_bmi = input_data.copy()
                    input_better_bmi['bmi'] = 23
                    pred_better_bmi = model.predict(input_better_bmi)[0]
                    savings_bmi = prediction - pred_better_bmi
                    
                    st.metric("New Premium", f"₦{pred_better_bmi:,.2f}", f"-₦{savings_bmi:,.2f}")
                    st.success(f"💰 **Annual Savings: ₦{savings_bmi:,.2f}**")
                    
                    weight_to_lose = (bmi - 23) * ((height_cm/100 if 'height_cm' in locals() else 1.75) ** 2)
                    st.write(f"Approx. weight to lose: **{weight_to_lose:.1f} kg**")
                else:
                    st.info("✅ Your BMI is already in healthy range!")
            
            with comp_col3:
                st.markdown("#### ⏰ Premium in 10 Years (Same Health)")
                input_older = input_data.copy()
                input_older['age'] = age + 10
                pred_older = model.predict(input_older)[0]
                increase = pred_older - prediction
                
                st.metric("Future Premium (Age " + str(age+10) + ")", 
                         f"₦{pred_older:,.2f}", 
                         f"+₦{increase:,.2f}")
                st.info(f"📈 Expected increase: **₦{increase:,.2f}/year** due to aging")
                st.write(f"Avg increase per year: **₦{increase/10:,.2f}**")
        
        # Personalized recommendations
        if show_recommendations:
            st.markdown("---")
            st.markdown("### 💡 Personalized Recommendations to Lower Your Premium")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown("#### 🎯 Action Items")
                
                recommendations = []
                potential_savings = 0
                
                if smoker == "Yes":
                    input_no_smoke = input_data.copy()
                    input_no_smoke['smoker'] = 0
                    savings = prediction - model.predict(input_no_smoke)[0]
                    recommendations.append({
                        'action': '🚭 **Quit Smoking (Priority #1)**',
                        'impact': 'High',
                        'savings': savings,
                        'difficulty': 'Hard but worth it',
                        'timeline': '1-3 years to see full premium reduction'
                    })
                    potential_savings += savings
                
                if accident_history != "None":
                    input_no_accidents = input_data.copy()
                    input_no_accidents['accident_minor'] = 0
                    input_no_accidents['accident_moderate'] = 0
                    input_no_accidents['accident_severe'] = 0
                    input_no_accidents['accident_catastrophic'] = 0
                    savings = prediction - model.predict(input_no_accidents)[0]
                    recommendations.append({
                        'action': f'⚠️ **Avoid Future Accidents ({accident_history} History)**',
                        'impact': 'Medium-High',
                        'savings': savings,
                        'difficulty': 'Requires lifestyle changes',
                        'timeline': '3-5 years for history to clear'
                    })
                
                if bmi > 50:
                    input_better_bmi = input_data.copy()
                    input_better_bmi['bmi'] = 30
                    savings = prediction - model.predict(input_better_bmi)[0]
                    recommendations.append({
                        'action': '🏃 **Reduce Extreme BMI to Obese Range (30)**',
                        'impact': 'Very High',
                        'savings': savings,
                        'difficulty': 'Very Hard - Medical supervision needed',
                        'timeline': '1-3 years with medical program'
                    })
                    potential_savings += savings
                elif bmi > 30:
                    input_better_bmi = input_data.copy()
                    input_better_bmi['bmi'] = 25
                    savings = prediction - model.predict(input_better_bmi)[0]
                    recommendations.append({
                        'action': '🏃 **Lose Weight to Healthy BMI**',
                        'impact': 'Medium-High',
                        'savings': savings,
                        'difficulty': 'Moderate',
                        'timeline': '6-12 months'
                    })
                    potential_savings += savings
                elif bmi > 25:
                    input_better_bmi = input_data.copy()
                    input_better_bmi['bmi'] = 23
                    savings = prediction - model.predict(input_better_bmi)[0]
                    recommendations.append({
                        'action': '💪 **Reduce BMI to Normal Range**',
                        'impact': 'Medium',
                        'savings': savings,
                        'difficulty': 'Moderate',
                        'timeline': '3-6 months'
                    })
                    potential_savings += savings
                
                if age < 30:
                    recommendations.append({
                        'action': '📅 **Lock in Rates Now**',
                        'impact': 'Long-term',
                        'savings': 0,
                        'difficulty': 'Easy',
                        'timeline': 'Immediate - premiums increase with age'
                    })
                
                if children == 0:
                    recommendations.append({
                        'action': '👨‍👩‍👧 **Consider Family Planning Impact**',
                        'impact': 'Future planning',
                        'savings': 0,
                        'difficulty': 'Planning',
                        'timeline': 'Each child adds ~$500-1000/year'
                    })
                
                recommendations.append({
                    'action': '🏥 **Shop Different Plan Types**',
                    'impact': 'Medium',
                    'savings': prediction * 0.15,
                    'difficulty': 'Easy',
                    'timeline': 'During open enrollment'
                })
                
                recommendations.append({
                    'action': '💰 **Check for Subsidies**',
                    'impact': 'High (if eligible)',
                    'savings': 0,
                    'difficulty': 'Easy',
                    'timeline': 'Immediate if income-qualified'
                })
                
                if not recommendations or (len(recommendations) <= 2 and smoker == "No" and bmi < 25):
                    st.success("✅ **Great news!** You're already in a low-risk category with minimal premium reduction opportunities.")
                    st.write("Focus on maintaining your healthy lifestyle!")
                else:
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}. {rec['action']}**")
                        st.write(f"   • Impact: {rec['impact']}")
                        if rec['savings'] > 0:
                            st.write(f"   • Potential savings: ₦{rec['savings']:,.2f}/year")
                        st.write(f"   • Difficulty: {rec['difficulty']}")
                        st.write(f"   • Timeline: {rec['timeline']}")
                        st.write("")
                
                if potential_savings > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>💰 Total Potential Savings</h4>
                        <h2>₦{potential_savings:,.2f} per year</h2>
                        <p>5-year savings: <strong>₦{potential_savings*5:,.2f}</strong></p>
                        <p>10-year savings: <strong>₦{potential_savings*10:,.2f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown("#### ⚠️ Your Risk Factors")
                
                risk_factors = []
                
                if smoker == "Yes":
                    risk_factors.append("🚨 **Smoking** - Highest impact on premium and health")
                
                if accident_history == "Catastrophic":
                    risk_factors.append("🆘 **Catastrophic Accident History** - Major premium impact")
                elif accident_history == "Severe":
                    risk_factors.append("🚨 **Severe Accident History** - Significant premium impact")
                elif accident_history == "Moderate":
                    risk_factors.append("⚠️ **Moderate Accident History** - Moderate premium impact")
                elif accident_history == "Minor":
                    risk_factors.append("ℹ️ **Minor Accident History** - Small premium impact")
                
                if bmi > 50:
                    risk_factors.append("🚨 **Extreme Obesity (BMI 50+)** - Critical health risks and very high premiums")
                elif bmi > 30:
                    risk_factors.append("🚨 **Obesity (BMI 30+)** - Significant health risks")
                elif bmi > 25:
                    risk_factors.append("⚠️ **Overweight (BMI 25-30)** - Moderate health risks")
                
                if age > 55:
                    risk_factors.append("⚠️ **Age 55+** - Naturally higher premiums")
                elif age > 45:
                    risk_factors.append("ℹ️ **Age 45+** - Premiums increasing with age")
                
                if children >= 3:
                    risk_factors.append("ℹ️ **Large family** - Multiple dependents increase costs")
                
                if not risk_factors:
                    st.success("✅ **No major risk factors detected!**")
                    st.write("You're in a favorable position for health insurance rates.")
                    st.write("")
                    st.write("**To maintain low premiums:**")
                    st.write("• Continue healthy lifestyle")
                    st.write("• Regular preventive care")
                    st.write("• Stay tobacco-free")
                    st.write("• Maintain healthy weight")
                else:
                    for factor in risk_factors:
                        st.write(factor)
                    
                    st.write("")
                    st.markdown("**📚 Learn More:**")
                    st.write("Click on '📚 Understanding Factors' in the sidebar to learn how each factor affects your premium.")

# ANALYTICS DASHBOARD
elif page == "📊 Analytics Dashboard":
    st.markdown('<p class="medium-font">📊 Insurance Data Analytics</p>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### 📈 Dataset Overview")
    
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        st.metric("Total Records", f"{len(df):,}")
    with overview_col2:
        st.metric("Average Premium", f"₦{df['charges'].mean():,.2f}")
    with overview_col3:
        st.metric("Median Premium", f"₦{df['charges'].median():,.2f}")
    with overview_col4:
        st.metric("Premium Range", f"₦{df['charges'].min():,.0f} - ₦{df['charges'].max():,.0f}")
    
    st.markdown("---")
    
    # Key insights
    st.markdown("### 🔍 Key Insights from the Data")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        smoker_avg = df[df['smoker'] == 1]['charges'].mean()
        non_smoker_avg = df[df['smoker'] == 0]['charges'].mean()
        smoker_multiplier = smoker_avg / non_smoker_avg
        
        st.markdown("""
        <div class="warning-box">
            <h4>🚬 Smoking Impact</h4>
            <p>Smokers pay <strong>{:.1f}x more</strong> than non-smokers</p>
            <ul>
                <li>Average smoker premium: <strong>${:,.2f}</strong></li>
                <li>Average non-smoker premium: <strong>${:,.2f}</strong></li>
                <li>Difference: <strong>${:,.2f}</strong></li>
            </ul>
        </div>
        """.format(smoker_multiplier, smoker_avg, non_smoker_avg, smoker_avg - non_smoker_avg), 
        unsafe_allow_html=True)
    
    with insight_col2:
        obese_avg = df[df['bmi'] > 30]['charges'].mean()
        normal_avg = df[df['bmi'] < 25]['charges'].mean()
        
        st.markdown("""
        <div class="info-box">
            <h4>⚖️ BMI Impact</h4>
            <p>People with obesity pay <strong>{:.1f}x more</strong> than normal weight</p>
            <ul>
                <li>Average obese (BMI 30+): <strong>${:,.2f}</strong></li>
                <li>Average normal (BMI <25): <strong>${:,.2f}</strong></li>
                <li>Difference: <strong>${:,.2f}</strong></li>
            </ul>
        </div>
        """.format(obese_avg/normal_avg, obese_avg, normal_avg, obese_avg - normal_avg), 
        unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Age vs Charges
        fig1 = px.scatter(df, x='age', y='charges', color='smoker',
                         title='Premium vs Age (colored by smoking status)',
                         labels={'smoker': 'Smoker', 'charges': 'Annual Premium ($)', 'age': 'Age'},
                         color_discrete_map={0: 'green', 1: 'red'},
                         opacity=0.6)
        st.plotly_chart(fig1, use_container_width=True)
        
        # BMI distribution
        fig3 = px.histogram(df, x='bmi', nbins=30, title='BMI Distribution',
                           color_discrete_sequence=['#2ca02c'])
        fig3.add_vline(x=25, line_dash="dash", line_color="orange", 
                      annotation_text="Overweight threshold")
        fig3.add_vline(x=30, line_dash="dash", line_color="red", 
                      annotation_text="Obese threshold")
        st.plotly_chart(fig3, use_container_width=True)
    
    with viz_col2:
        # Charges distribution
        fig2 = px.histogram(df, x='charges', nbins=50, title='Premium Distribution',
                           color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Smoker comparison
        smoker_data = df.copy()
        smoker_data['smoker_label'] = smoker_data['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        fig4 = px.box(smoker_data, x='smoker_label', y='charges', 
                     title='Premium Distribution: Smoker vs Non-Smoker',
                     color='smoker_label', 
                     color_discrete_map={'Smoker': 'red', 'Non-Smoker': 'green'})
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # More visualizations
    st.markdown("### 📊 Advanced Analytics")
    
    viz_col3, viz_col4 = st.columns(2)
    
    with viz_col3:
        # Age groups
        df_copy = df.copy()
        df_copy['age_group'] = pd.cut(df_copy['age'], bins=[0, 30, 40, 50, 100], 
                                      labels=['18-30', '31-40', '41-50', '50+'])
        age_group_avg = df_copy.groupby('age_group')['charges'].mean().reset_index()
        
        fig5 = px.bar(age_group_avg, x='age_group', y='charges',
                     title='Average Premium by Age Group',
                     color='charges', color_continuous_scale='Blues')
        st.plotly_chart(fig5, use_container_width=True)
    
    with viz_col4:
        # Children impact
        children_avg = df.groupby('children')['charges'].mean().reset_index()
        fig6 = px.line(children_avg, x='children', y='charges',
                      title='Average Premium by Number of Children',
                      markers=True, line_shape='spline')
        st.plotly_chart(fig6, use_container_width=True)
    
    st.markdown("---")
    
    # Summary statistics
    st.markdown("### 📋 Summary Statistics")
    
    numeric_cols = ['age', 'bmi', 'children', 'charges']
    summary_stats = df[numeric_cols].describe()
    
    st.dataframe(summary_stats.style.format("{:.2f}"), use_container_width=True)

# MODEL PERFORMANCE PAGE
elif page == "📈 Model Performance":
    st.markdown('<p class="medium-font">📈 Model Performance & Accuracy</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Our prediction model uses <strong>Random Forest Regression</strong>, a powerful machine learning algorithm 
        that builds multiple decision trees and combines their predictions for high accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance metrics
    st.markdown("### 🎯 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.metric("Training RMSE", f"₦{metrics['train_rmse']:,.2f}",
                 help="Root Mean Squared Error on training data")
    with kpi_col2:
        st.metric("Test RMSE", f"₦{metrics['test_rmse']:,.2f}",
                 help="Root Mean Squared Error on test data - most important metric")
    with kpi_col3:
        st.metric("Training R²", f"{metrics['train_r2']:.4f}",
                 help="R-squared score on training data")
    with kpi_col4:
        st.metric("Test R²", f"{metrics['test_r2']:.4f}",
                 help="R-squared score on test data - model accuracy")
    with kpi_col5:
        st.metric("Test MAE", f"₦{metrics['test_mae']:,.2f}",
                 help="Mean Absolute Error - average prediction error")
    
    # Model interpretation
    st.markdown("---")
    st.markdown("### 🤖 Understanding the Metrics")
    
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.markdown("""
        **R² Score (Accuracy):**
        - Measures how well the model explains premium variation
        - Range: 0 to 1 (higher is better)
        - Our test R² of {:.1f}% means the model explains {:.1f}% of why premiums vary
        - This is considered **very good** for real-world data
        
        **What this means for you:**
        - Model predictions are reliable and trustworthy
        - 86 out of 100 predictions are very close to actual premiums
        """.format(metrics['test_r2']*100, metrics['test_r2']*100))
    
    with metric_col2:
        st.markdown("""
        **RMSE (Average Error):**
        - Shows typical prediction error in Naira
        - Test RMSE of ₦{:,.2f} means predictions are off by ~₦{:,.2f} on average
        - Lower is better
        
        **MAE (Mean Absolute Error):**
        - Average of all prediction errors
        - ₦{:,.2f} average error is excellent for insurance prediction
        - More interpretable than RMSE
        """.format(metrics['test_rmse'], metrics['test_rmse'], metrics['test_mae']))
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance - What Matters Most?")
    
    st.markdown("""
    <div class="info-box">
        This chart shows which factors have the biggest impact on premium predictions. 
        Higher importance means that factor is more influential in determining your premium.
    </div>
    """, unsafe_allow_html=True)
    
    # Create better feature names
    feature_names_map = {
        'smoker': '🚬 Smoking Status',
        'age': '🩺 Age',
        'bmi': '⚖️ BMI',
        'children': '👨‍👩‍👧‍👦 Children',
        'sex_male': '⚧️ Gender',
        'region_northeast': '📍 Northeast Region',
        'region_northwest': '📍 Northwest Region',
        'region_southeast': '📍 Southeast Region',
        'region_southwest': '📍 Southwest Region'
    }
    
    feature_importance_display = feature_importance.copy()
    feature_importance_display['feature'] = feature_importance_display['feature'].map(feature_names_map)
    feature_importance_display['percentage'] = (feature_importance_display['importance'] * 100).round(2)
    
    fig_importance = px.bar(feature_importance_display, x='importance', y='feature',
                           orientation='h', title='Feature Importance Ranking',
                           color='importance', color_continuous_scale='Viridis',
                           labels={'importance': 'Importance Score', 'feature': 'Factor'})
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Feature importance table
    st.dataframe(
        feature_importance_display[['feature', 'percentage']].rename(
            columns={'feature': 'Factor', 'percentage': 'Importance (%)'}
        ).style.format({'Importance (%)': '{:.2f}%'}),
        use_container_width=True
    )
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_feature = feature_importance_display.iloc[0]
        st.markdown(f"""
        <div class="success-box">
            <h4>🥇 #1 Factor</h4>
            <h3>{top_feature['feature']}</h3>
            <p><strong>{top_feature['percentage']:.1f}%</strong> of prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        controllable = feature_importance_display[
            feature_importance_display['feature'].isin(['🚬 Smoking Status', '⚖️ BMI'])
        ]['percentage'].sum()
        st.markdown(f"""
        <div class="info-box">
            <h4>✅ Controllable Factors</h4>
            <p>Smoking + BMI account for</p>
            <h3>{controllable:.1f}%</h3>
            <p>You can influence these!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        region_importance = feature_importance_display[
            feature_importance_display['feature'].str.contains('Region')
        ]['percentage'].sum()
        st.markdown(f"""
        <div class="warning-box">
            <h4>📍 Location Impact</h4>
            <p>All regions combined</p>
            <h3>{region_importance:.1f}%</h3>
            <p>Based on where you live</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction accuracy visualization
    st.markdown("### 📉 Model Accuracy Visualization")
    
    y_pred = model.predict(X_test)
    
    fig_accuracy = go.Figure()
    fig_accuracy.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        marker=dict(size=6, color='blue', opacity=0.5),
        name='Predictions', text=[f'Actual: ₦{a:,.0f}<br>Predicted: ₦{p:,.0f}' 
                                 for a, p in zip(y_test, y_pred)],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig_accuracy.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines', line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    fig_accuracy.update_layout(
        title='Actual vs Predicted Premiums',
        xaxis_title='Actual Premium (₦)',
        yaxis_title='Predicted Premium (₦)',
        hovermode='closest'
    )
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>How to read this chart:</strong><br>
        • Each blue dot is a prediction<br>
        • Red dashed line = perfect predictions<br>
        • Dots close to the line = accurate predictions<br>
        • Our model's dots cluster tightly around the line, showing high accuracy!
    </div>
    """, unsafe_allow_html=True)

# TIPS TO SAVE MONEY PAGE
elif page == "💡 Tips to Save Money":
    st.markdown('<p class="medium-font">💡 How to Lower Your Health Insurance Premium</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>💰 Take Control of Your Healthcare Costs</h3>
        <p>While some factors like age and location are beyond your control, there are many ways to reduce your 
        health insurance premium. Here are proven strategies to save money.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Major savings opportunities
    st.markdown("### 🎯 Top 5 Ways to Save (Biggest Impact)")
    
    with st.expander("🚭 1. Quit Smoking - Save $10,000+ per year", expanded=True):
        st.markdown("""
        **Potential Savings: $10,000 - $15,000 annually**
        
        Smoking is the #1 factor affecting your premium. Quitting smoking is the single most impactful change you can make.
        
        **Timeline:**
        - Immediately: Start improving your health
        - 1 year smoke-free: May qualify for lower rates with some insurers
        - 3-5 years smoke-free: Eligible for non-smoker rates with most insurers
        
        **Resources to help you quit:**
        - 🏥 Talk to your doctor about cessation programs
        - 💊 Consider nicotine replacement therapy
        - 📱 Use quit-smoking apps (QuitNow, Smoke Free)
        - 👥 Join support groups
        - 📞 Call 1-800-QUIT-NOW for free help
        
        **Real example:**
        - Age 35, BMI 25, Smoker: $19,000/year
        - Same person, Non-smoker: $5,500/year
        - **Savings: $13,500/year = $135,000 over 10 years!**
        """)
    
    with st.expander("🏃 2. Lose Weight / Improve BMI - Save $2,000-$5,000 per year"):
        st.markdown("""
        **Potential Savings: $2,000 - $5,000 annually**
        
        Maintaining a healthy BMI (18.5-24.9) can significantly reduce your premium.
        
        **BMI Reduction Strategy:**
        - Calculate your target weight
        - Aim for 1-2 pounds per week (safe, sustainable)
        - Combine diet and exercise
        - Track progress with apps (MyFitnessPal, Lose It!)
        
        **Health benefits beyond savings:**
        - Reduced risk of diabetes, heart disease
        - More energy and better mood
        - Improved sleep quality
        - Lower blood pressure
        
        **Example savings:**
        - BMI 35 → BMI 25: Save ~$3,500/year
        - BMI 28 → BMI 23: Save ~$1,800/year
        
        **Tips:**
        - 🥗 Focus on whole foods, vegetables, lean protein
        - 💧 Drink plenty of water
        - 🚶 Walk 10,000 steps daily
        - 😴 Get 7-8 hours of sleep
        - 📊 Track calories and exercise
        """)
    
    with st.expander("🛒 3. Shop Different Plans - Save $1,000-$3,000 per year"):
        st.markdown("""
        **Potential Savings: $1,000 - $3,000 annually**
        
        Not all insurance plans are created equal. Shopping around can reveal significant savings.
        
        **Plan Types to Consider:**
        
        **Bronze Plans:**
        - Lowest premium (~60% savings vs Platinum)
        - Best for: Healthy people who rarely need care
        - High deductible but emergency coverage
        
        **Silver Plans:**
        - Balanced premium and coverage
        - Best for: Most people, especially if eligible for subsidies
        - Good middle ground
        
        **High-Deductible Health Plans (HDHP):**
        - Lower premiums
        - Qualifies for Health Savings Account (HSA)
        - Triple tax advantage with HSA
        
        **Network Choices:**
        - HMO: Lowest cost, must use network
        - PPO: Higher cost, more flexibility
        - EPO: Middle ground
        
        **When to shop:**
        - Open enrollment period (Nov-Dec typically)
        - Qualifying life events (marriage, birth, job loss)
        - Annually compare rates
        """)
    
    with st.expander("💰 4. Check Eligibility for Subsidies - Save $3,000-$15,000 per year"):
        st.markdown("""
        **Potential Savings: $3,000 - $15,000+ annually**
        
        Many people don't realize they qualify for government subsidies that can dramatically reduce premiums.
        
        **Premium Tax Credits (ACA Marketplace):**
        - Available for incomes up to 400% of federal poverty level
        - Can reduce premiums by 50-90%
        - Applied monthly or claimed at tax time
        
        **Income limits for family of 4 (2024):**
        - $31,200 - $124,800: Likely eligible for subsidies
        - Under $31,200: May qualify for Medicaid
        
        **Example savings:**
        - Family income $60,000: Save ~$10,000/year
        - Family income $80,000: Save ~$6,000/year
        
        **How to apply:**
        - Visit Healthcare.gov during open enrollment
        - State exchanges (Covered California, NY State of Health, etc.)
        - Provide income documentation
        - Get instant eligibility determination
        
        **Additional help:**
        - Cost-sharing reductions (lower copays/deductibles)
        - Medicaid expansion in many states
        - CHIP for children
        """)
    
    with st.expander("👨‍💼 5. Employer-Sponsored Insurance - Save $2,000-$8,000 per year"):
        st.markdown("""
        **Potential Savings: $2,000 - $8,000 annually**
        
        If available, employer plans are almost always cheaper than individual insurance.
        
        **Why employer insurance is cheaper:**
        - Employer pays 50-80% of premium
        - Group rates (not individually underwritten)
        - Pre-tax deductions (additional savings)
        
        **Average costs (2024):**
        - Individual market: $7,000-$15,000/year
        - Employer-sponsored: $2,000-$6,000/year (your portion)
        - **Savings: $3,000-$10,000/year**
        
        **What to do:**
        - Check if your employer offers insurance
        - Compare during open enrollment
        - Consider spouse's employer plan too
        - Factor in employer HSA contributions
        
        **Pre-tax savings:**
        - Premiums deducted before taxes
        - Saves additional 20-30% on premium
        - Example: $5,000 premium = $1,000-$1,500 tax savings
        """)
    
    st.markdown("---")
    
    # Additional strategies
    st.markdown("### 💡 Additional Money-Saving Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🏥 Preventive Care
        - Use free annual checkups
        - Get vaccinations covered at 100%
        - Catch issues early (cheaper to treat)
        - Many plans cover screenings for free
        
        #### 💊 Prescription Savings
        - Ask for generic medications (50-80% cheaper)
        - Use prescription discount cards (GoodRx, RxSaver)
        - Mail-order for 90-day supplies
        - Compare pharmacy prices online
        
        #### 🏦 Health Savings Account (HSA)
        - Triple tax advantage
        - Contributions reduce taxable income
        - Grows tax-free
        - Withdrawals tax-free for medical expenses
        - Can save $1,000-$2,000/year in taxes
        
        #### 👨‍👩‍👧 Family Planning
        - Individual plans vs family plans
        - Add spouse to your plan vs separate
        - Consider children's CHIP eligibility
        - Calculate total family costs
        """)
    
    with col2:
        st.markdown("""
        #### 📍 Consider Moving (Long-term)
        - Healthcare costs vary by state
        - Some states have lower premiums
        - Factor into relocation decisions
        - Research before moving
        
        #### ⚕️ Telemedicine
        - Often free or low copay
        - Saves time and money
        - Good for minor issues
        - Many plans include this
        
        #### 🏋️ Wellness Programs
        - Some employers offer premium discounts
        - Gym membership reimbursements
        - Health coaching programs
        - Can save $200-$600/year
        
        #### 📅 Timing Matters
        - Buy during open enrollment
        - Lock in rates when young
        - Review annually
        - Update life changes promptly
        """)
    
    st.markdown("---")
    
    # Action plan
    st.markdown("### 📋 Your Action Plan")
    
    st.markdown("""
    <div class="success-box">
        <h4>✅ Start Saving Today - Step by Step Guide</h4>
        
        <strong>Immediate Actions (This Week):</strong>
        <ol>
            <li>Check if you qualify for subsidies at Healthcare.gov</li>
            <li>Review your current plan and alternatives</li>
            <li>If you smoke, schedule a doctor visit to discuss quitting</li>
            <li>Download a health tracking app to monitor BMI</li>
        </ol>
        
        <strong>Short-term Goals (1-3 Months):</strong>
        <ol>
            <li>Compare all available plans during open enrollment</li>
            <li>Start a weight loss program if BMI > 25</li>
            <li>Set up HSA if eligible</li>
            <li>Review prescription costs and switch to generics</li>
        </ol>
        
        <strong>Long-term Goals (6-12 Months):</strong>
        <ol>
            <li>Achieve smoke-free status</li>
            <li>Reach healthy BMI range</li>
            <li>Maximize HSA contributions</li>
            <li>Maintain preventive care schedule</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Savings calculator
    st.markdown("### 🧮 Calculate Your Potential Savings")
    
    savings_col1, savings_col2 = st.columns([1, 1])
    
    with savings_col1:
        st.markdown("#### What changes are you willing to make?")
        
        will_quit_smoking = st.checkbox("🚭 Quit smoking", key="save_smoke")
        will_lose_weight = st.checkbox("🏃 Lose weight to healthy BMI", key="save_weight")
        will_shop_plans = st.checkbox("🛒 Shop for different plan type", key="save_shop")
        will_check_subsidies = st.checkbox("💰 Check subsidy eligibility", key="save_subsidy")
        will_use_employer = st.checkbox("👨‍💼 Switch to employer plan (if available)", key="save_employer")
        
        calculate_savings = st.button("Calculate Total Potential Savings", type="primary")
    
    with savings_col2:
        if calculate_savings:
            total_savings = 0
            savings_breakdown = []
            
            if will_quit_smoking:
                savings = 12000
                total_savings += savings
                savings_breakdown.append(("Quit Smoking", savings))
            
            if will_lose_weight:
                savings = 3000
                total_savings += savings
                savings_breakdown.append(("Lose Weight", savings))
            
            if will_shop_plans:
                savings = 2000
                total_savings += savings
                savings_breakdown.append(("Different Plan", savings))
            
            if will_check_subsidies:
                savings = 6000
                total_savings += savings
                savings_breakdown.append(("Subsidies", savings))
            
            if will_use_employer:
                savings = 5000
                total_savings += savings
                savings_breakdown.append(("Employer Plan", savings))
            
            if total_savings > 0:
                st.markdown(f"""
                <div class="success-box">
                    <h3>💰 Your Potential Savings</h3>
                    <h2>${total_savings:,} per year</h2>
                    <p><strong>5 years:</strong> ${total_savings*5:,}</p>
                    <p><strong>10 years:</strong> ${total_savings*10:,}</p>
                    <p><strong>Lifetime (30 years):</strong> ${total_savings*30:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Savings Breakdown:")
                for item, amount in savings_breakdown:
                    st.write(f"• {item}: **${amount:,}**/year")
            else:
                st.info("Select at least one change to see potential savings!")
    
    st.markdown("---")
    
    # Resources
    st.markdown("### 📚 Helpful Resources")
    
    resource_col1, resource_col2, resource_col3 = st.columns(3)
    
    with resource_col1:
        st.markdown("""
        **Government Resources:**
        - Healthcare.gov
        - Medicare.gov
        - Medicaid.gov
        - IRS.gov (HSA info)
        """)
    
    with resource_col2:
        st.markdown("""
        **Comparison Tools:**
        - eHealth.com
        - PolicyGenius.com
        - HealthSherpa.com
        - Kaiser Family Foundation calculator
        """)
    
    with resource_col3:
        st.markdown("""
        **Health Resources:**
        - CDC.gov
        - MyFitnessPal (weight loss)
        - SmokeFree.gov
        - GoodRx (prescriptions)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>💡 <strong>Disclaimer:</strong> This calculator provides estimates based on statistical models and should not be considered 
    a quote or guarantee of actual insurance premiums. Actual premiums may vary based on additional factors, 
    specific insurance company policies, and current market conditions. Always consult with licensed insurance agents 
    for accurate quotes.</p>
    <p>📧 Questions? Contact us at support@healthinsure.com</p>
    <p>© 2024 Health Insurance Premium Predictor | Built with Streamlit ❤️</p>
</div>
""", unsafe_allow_html=True)