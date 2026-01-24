import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.models.recommendation_engine import RecommendationEngine

# Page Config
st.set_page_config(
    page_title="ML Service Recommender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Engine
@st.cache_resource
def get_engine():
    return RecommendationEngine()

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Error loading models. Details: {e}")
    st.stop()

# Tailyo Technologies Color Palette & Custom CSS
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Main container */
    .main {
        background: #F8F9FA;
        padding: 0;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1A2332;
        font-weight: 600;
    }
    
    /* Primary Button */
    .stButton>button {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px 28px;
        font-size: 15px;
        font-weight: 600;
        height: 48px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(30, 136, 229, 0.25);
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        box-shadow: 0 4px 10px rgba(30, 136, 229, 0.35);
        transform: translateY(-1px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Input Fields */
    .stSelectbox, .stMultiSelect, .stTextInput {
        margin-bottom: 16px;
    }
    
    .stSelectbox label, .stMultiSelect label, .stTextInput label {
        font-weight: 500;
        color: #1A2332;
        font-size: 14px;
    }
    
    /* Radio buttons */
    .stRadio > div {
        flex-direction: row;
        gap: 12px;
    }
    
    /* Info boxes */
    .stInfo {
        background: #E8F4F8;
        border-left: 3px solid #1E88E5;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 14px;
    }
    
    /* Divider */
    hr {
        margin: 20px 0;
        border: none;
        height: 1px;
        background: #E1E4E8;
    }
</style>
""", unsafe_allow_html=True)

# Header Section with Logo
st.write("")
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 14px;'>
        <div style='width: 44px; height: 44px; background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); 
                    border-radius: 8px; display: flex; align-items: center; justify-content: center;'>
            <span style='color: white; font-size: 22px; font-weight: 700;'>T</span>
        </div>
        <div>
            <div style='font-size: 19px; font-weight: 600; color: #1A2332;'>Tailyo Technologies</div>
            <div style='font-size: 11px; color: #6C757D; letter-spacing: 0.3px;'>Service Recommendation System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 14px 0 24px 0;'>", unsafe_allow_html=True)

# Hero Section - Compact
st.markdown("""
<div style='background: linear-gradient(135deg, #EBF4FB 0%, #FFFFFF 100%); 
            padding: 20px 16px; 
            text-align: center; 
            margin-bottom: 20px;
            border-radius: 8px;
            border: 1px solid #D6E9F7;'>
    <h1 style='font-size: 26px; font-weight: 700; color: #0D47A1; margin-bottom: 6px;'>
        ML-Powered Service Recommendations
    </h1>
    <p style='font-size: 15px; color: #6C757D; max-width: 650px; margin: 0 auto;'>
        Find the perfect services for your business in seconds
    </p>
</div>
""", unsafe_allow_html=True)

# Main Input Form
col_left, col_center, col_right = st.columns([1, 2.5, 1])

with col_center:
    st.markdown("""
    <div style='background: white; 
                padding: 18px; 
                border-radius: 8px; 
                box-shadow: 0 2px 6px rgba(0,0,0,0.06);
                border: 1px solid #E1E4E8;
                margin-bottom: 16px;'>
        <h2 style='color: #1E88E5; font-size: 18px; font-weight: 600; text-align: center; margin-bottom: 12px;'>
            Tell Us Your Requirements
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Location Preference
    st.markdown("**Location Preference**")
    location_pref = st.radio(
        "Choose your location preference", 
        ["Remote", "Specific City"], 
        label_visibility="collapsed", 
        key="loc_pref", 
        horizontal=True
    )
    
    if location_pref == "Specific City":
        location_city = st.selectbox(
            "Select City", 
            ["Delhi", "Mumbai", "Bengaluru", "Chennai"], 
            key="city_select"
        )
        location = location_city.lower()
    else:
        location = "remote"
        st.caption("Services available remotely will be prioritized")
    
    st.write("")
    
    with st.form("preferences_form"):
        st.markdown("**Business Type**")
        business_types = ['E-commerce', 'Restaurant', 'Tech Startup', 'Retail', 'Freelancer', 'Clinic']
        target_business = st.selectbox(
            "Select your business category", 
            business_types, 
            label_visibility="collapsed"
        )
        
        st.markdown("**Budget Range**")
        budget = st.select_slider(
            "Budget Range", 
            options=['Low', 'Medium', 'High', 'Premium'], 
            value='Medium', 
            label_visibility="collapsed"
        )
        
        st.markdown("**Preferred Languages**")
        language_opt = st.radio(
            "Preferred Language", 
            ["English", "Hindi", "Both"], 
            horizontal=True,
            label_visibility="collapsed"
        )
        languages = [language_opt]
        
        st.markdown("**What Service Are You Looking For?**")
        service_options = sorted(engine.df['Service_Name'].unique())
        selected_service = st.selectbox(
            "Service Category", 
            service_options, 
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button(
            "Find Matching Services", 
            use_container_width=True
        )

# Results Section
if submitted:
    service_desc = engine.df[engine.df['Service_Name'] == selected_service]['Description'].iloc[0]
    
    user_input = {
        'Target_Business_Type': target_business,
        'Price_Category': budget,
        'Location_Area': location,
        'Language_Support': languages,
        'Description': service_desc
    }
    
    with st.spinner("Analyzing your preferences..."):
        results = engine.get_recommendations(user_input, top_k=3)
    
    if not results:
        st.markdown("""
        <div style='background: #FFF8E6; 
                    padding: 36px; 
                    border-radius: 8px; 
                    border-left: 3px solid #FFA726; 
                    text-align: center;
                    margin-top: 32px;'>
            <h3 style='color: #E65100; margin-bottom: 10px; font-size: 20px;'>No Matching Services Found</h3>
            <p style='color: #6C757D; margin-bottom: 20px; font-size: 14px;'>
                Try adjusting your preferences or broadening your search criteria
            </p>
            <div style='text-align: left; max-width: 380px; margin: 0 auto; color: #6C757D; font-size: 13px;'>
                <p style='margin-bottom: 6px; font-weight: 600;'>Suggestions:</p>
                <p style='margin: 4px 0;'>• Try selecting a different business type</p>
                <p style='margin: 4px 0;'>• Expand your budget range</p>
                <p style='margin: 4px 0;'>• Choose different language options</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='margin-top: 20px; padding-bottom: 12px; border-bottom: 1px solid #E1E4E8;'>
            <h2 style='color: #0D47A1; font-size: 20px; font-weight: 600; margin-bottom: 3px;'>
                Top Recommended Services
            </h2>
            <p style='color: #6C757D; font-size: 13px;'>
                Based on your preferences
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(results) < 5:
            st.info(f"Found {len(results)} service(s) matching all your criteria.")
        
        st.write("")
        
        # Create Amazon-style carousel cards HTML
        cards_html = '''
        <style>
            * {
                box-sizing: border-box;
            }
            .carousel-wrapper {
                position: relative;
                max-width: 100%;
                margin: 0 auto;
                padding: 0 60px;
            }
            .carousel-container {
                display: flex;
                overflow-x: auto;
                overflow-y: hidden;
                gap: 20px;
                padding: 24px 12px 64px 12px;
                scroll-behavior: smooth;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: thin;
                scrollbar-color: #1E88E5 #F0F2F5;
                justify-content: center;
            }
            .carousel-container::-webkit-scrollbar {
                height: 12px;
            }
            .carousel-container::-webkit-scrollbar-track {
                background: #F0F2F5;
                border-radius: 6px;
                margin: 0 12px;
            }
            .carousel-container::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
                border-radius: 6px;
                border: 2px solid #F0F2F5;
            }
            .carousel-container::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
            }
            .service-card {
                min-width: 290px;
                max-width: 290px;
                background: linear-gradient(to bottom, #FFFFFF 0%, #FAFBFC 100%);
                border: 2px solid #DFE3E8;
                border-radius: 12px;
                padding: 18px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.08);
                transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
                flex-shrink: 0;
                position: relative;
                overflow: hidden;
            }
            .service-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #1E88E5 0%, #1565C0 50%, #0D47A1 100%);
                opacity: 0;
                transition: opacity 0.35s ease;
            }
            .service-card:hover::before {
                opacity: 1;
            }
            .service-card:hover {
                border-color: #1E88E5;
                box-shadow: 0 12px 28px rgba(30,136,229,0.18), 0 4px 12px rgba(30,136,229,0.12);
                transform: translateY(-6px) scale(1.02);
                background: #FFFFFF;
            }
        </style>
        <div class="carousel-wrapper">
            <div class="carousel-container" id="carouselContainer">
        '''
        
        for i, res in enumerate(results):
            score = res['Match_Score']
            
            if score >= 80:
                badge_color = "#4CAF50"
                badge_bg = "linear-gradient(135deg, #4CAF50 0%, #388E3C 100%)"
                badge_text = "HIGH MATCH"
            elif score >= 60:
                badge_color = "#2196F3"
                badge_bg = "linear-gradient(135deg, #2196F3 0%, #1976D2 100%)"
                badge_text = "GOOD MATCH"
            else:
                badge_color = "#FF9800"
                badge_bg = "linear-gradient(135deg, #FF9800 0%, #F57C00 100%)"
                badge_text = "ALTERNATIVE"
            
            explanations_html = ""
            if 'Explanations' in res and res['Explanations']:
                explanations_html = """
                <div style='margin-top: 14px; padding-top: 12px; border-top: 1px solid #E9ECEF;'>
                    <div style='font-weight: 600; font-size: 13px; color: #1A2332; margin-bottom: 8px; display: flex; align-items: center;'>
                        <span style='display: inline-block; width: 4px; height: 14px; background: linear-gradient(180deg, #1E88E5 0%, #0D47A1 100%); border-radius: 2px; margin-right: 8px;'></span>
                        Why This Match?
                    </div>
                    <ul style='margin: 0; padding-left: 20px; color: #495057; font-size: 12px; line-height: 1.6;'>
                """
                for exp in res['Explanations']:
                    explanations_html += f"<li style='margin: 4px 0;'>{exp}</li>"
                explanations_html += "</ul></div>"
            
            desc = res['Description']
            if len(desc) > 85:
                desc = desc[:85] + "..."
            
            cards_html += f"""
            <div class="service-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                    <div style="flex: 1;">
                        <div style="display: inline-block; background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                                    color: #1565C0; padding: 3px 10px; border-radius: 12px; font-size: 11px; 
                                    font-weight: 700; margin-bottom: 6px; border: 1px solid #90CAF9;">
                            #{i+1}
                        </div>
                        <h3 style="margin: 0; font-size: 17px; color: #0D47A1; font-weight: 700; line-height: 1.3; 
                                   letter-spacing: -0.3px;">
                            {res['Service_Name']}
                        </h3>
                        <div style="font-size: 11px; color: #6C757D; margin-top: 6px; font-weight: 500;">
                            {res.get('Target_Business_Type', 'General').title()} • {res['Location'].title()}
                        </div>
                    </div>
                </div>
                
                <div style="background: {badge_bg}; color: white; padding: 6px 12px; border-radius: 6px; 
                            font-size: 10px; font-weight: 700; display: inline-block; margin-bottom: 14px;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.15); letter-spacing: 0.5px; text-transform: uppercase;">
                    {badge_text}
                </div>
                
                <div style="margin: 14px 0;">
                    <div style="background: linear-gradient(to right, #F0F2F5 0%, #E9ECEF 100%); 
                                height: 8px; border-radius: 4px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="background: {badge_bg}; height: 100%; width: {score}%; 
                                    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); 
                                    box-shadow: 0 0 8px rgba(0,0,0,0.2);"></div>
                    </div>
                    <div style="font-weight: 700; margin-top: 6px; font-size: 14px; color: #1A2332;">
                        {score}% <span style="font-weight: 500; color: #6C757D; font-size: 12px;">Match Score</span>
                    </div>
                </div>
                
                <div style="font-size: 12px; color: #495057; line-height: 1.6; margin: 14px 0; min-height: 65px;
                            background: #F8F9FA; padding: 10px; border-radius: 6px; border-left: 3px solid {badge_color};">
                    {desc}
                </div>
                
                <div style="padding-top: 12px; margin-top: 12px; border-top: 2px solid #E9ECEF;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-weight: 700; font-size: 12px; color: #1A2332;">Price:</span> 
                        <span style="background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); 
                                     color: #E65100; font-size: 12px; font-weight: 600; padding: 4px 10px; 
                                     border-radius: 4px; border: 1px solid #FFB74D;">
                            {res['Price_Category'].title()}
                        </span>
                    </div>
                </div>
                
                {explanations_html}
            </div>
            """
        
        cards_html += '''
            </div>
        </div>
        '''
        
        import streamlit.components.v1 as components
        components.html(cards_html, height=600, scrolling=False)

else:
    # Features Section
    st.markdown("""
    <div style='margin-top: 48px; padding: 36px 20px; text-align: center;'>
        <div style='display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; max-width: 850px; margin: 0 auto;'>
            <div style='flex: 1; min-width: 180px;'>
                <div style='width: 56px; height: 56px; background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                            border-radius: 12px; display: flex; align-items: center; justify-content: center; 
                            margin: 0 auto 12px; border: 2px solid #1E88E5;'>
                    <span style='color: #1E88E5; font-size: 28px; font-weight: 700;'>AI</span>
                </div>
                <h3 style='color: #0D47A1; font-size: 16px; margin-bottom: 6px; font-weight: 600;'>AI-Powered</h3>
                <p style='color: #6C757D; font-size: 13px;'>Advanced machine learning algorithms</p>
            </div>
            <div style='flex: 1; min-width: 180px;'>
                <div style='width: 56px; height: 56px; background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); 
                            border-radius: 12px; display: flex; align-items: center; justify-content: center; 
                            margin: 0 auto 12px; border: 2px solid #FF9800;'>
                    <span style='color: #FF9800; font-size: 24px; font-weight: 700;'>⚡</span>
                </div>
                <h3 style='color: #0D47A1; font-size: 16px; margin-bottom: 6px; font-weight: 600;'>Instant Results</h3>
                <p style='color: #6C757D; font-size: 13px;'>Get recommendations in seconds</p>
            </div>
            <div style='flex: 1; min-width: 180px;'>
                <div style='width: 56px; height: 56px; background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); 
                            border-radius: 12px; display: flex; align-items: center; justify-content: center; 
                            margin: 0 auto 12px; border: 2px solid #4CAF50;'>
                    <span style='color: #4CAF50; font-size: 28px; font-weight: 700;'>✓</span>
                </div>
                <h3 style='color: #0D47A1; font-size: 16px; margin-bottom: 6px; font-weight: 600;'>Top Matches</h3>
                <p style='color: #6C757D; font-size: 13px;'>Curated for your business needs</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='margin-top: 48px; 
            padding: 24px 20px; 
            background: white; 
            border-top: 1px solid #E1E4E8; 
            text-align: center;'>
    <p style='color: #6C757D; font-size: 13px; margin-bottom: 6px;'>
        Powered by Machine Learning • KNN + Cosine Similarity • 100% Accuracy
    </p>
    <p style='color: #ADB5BD; font-size: 11px;'>
        © 2025 Tailyo Technologies. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)