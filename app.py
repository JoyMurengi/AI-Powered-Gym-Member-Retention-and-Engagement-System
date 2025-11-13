#
#
#
#
#
#
import streamlit as st
import pandas as pd
import joblib
import re
import string

# --- 1. NLP PREPROCESSING UTILITIES ---
# Improved stopwords list (syntax fixed) + numeric handling
COMMON_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
}

def preprocess_text(text):
    """Cleans and prepares text for topic analysis (numbers preserved)."""
    text = text.lower()
    # Replace hyphens with space for phrases like "29-year-old"
    text = text.replace("-", " ")
    # Remove punctuation except numbers and '+'
    text = re.sub(f"[{re.escape(string.punctuation.replace('+',''))}]", "", text)
    # Tokenize words/numbers
    words = re.findall(r'\b[a-z0-9]+\b', text)
    processed_words = [word for word in words if word not in COMMON_STOP_WORDS]
    return " ".join(processed_words)

# --- 2. Topic & Recommendation Logic ---
def determine_topic(processed_text):
    if not processed_text:
        return 'No Feedback', 0
    if any(word in processed_text for word in ['worth', 'money', 'disappointing', 'charges', 'high', 'paid', 'expensive']):
        return 'Value & Quality', 3
    if any(word in processed_text for word in ['decided', 'leave', 'negative', 'experience', 'frankly', 'contract', 'fit']):
        return 'Hard Churn Signal', 4
    if any(word in processed_text for word in ['never', 'settled', 'using', 'enough', 'routine']):
        return 'Usage / Integration Issue', 5
    if any(word in processed_text for word in ['excellent', 'recommend', 'love', 'attend', 'great', 'convenient', 'best']):
        return 'Ambassadors / High Engagement', 1
    return 'Neutral / Other', 0

def generate_recommendation(topic_name, churn_status, raw_feedback):
    if topic_name == 'Value & Quality':
        if churn_status == 'Left':
             return "Retention Team: **High-Priority Win-Back.** Offer a 3-month contract 'Freeze' or a 20% discount."
        else:
             return "Customer Success: **Flight Risk Alert.** Send service quality survey."
    elif topic_name == 'Hard Churn Signal':
        if churn_status == 'Left':
             return "High-Priority Sales: **Immediate Exit Interview.** Call customer to understand loss and offer 'Win-back' package."
        else:
             return "Customer Success: **Investigate Incident.** Customer expressing intent to leave."
    elif topic_name == 'Usage / Integration Issue':
        if churn_status == 'Left':
            return "Retention Team: **Targeted Re-engagement.** Offer free 1-hour consultation with a personal trainer."
        else:
             return "Customer Success: **Proactive Coaching.** Call to check attendance and class frequency."
    elif topic_name == 'Ambassadors / High Engagement':
        if 'recommend' in raw_feedback.lower():
            return "Marketing: **Capitalize on Loyalty.** Send 'Referral Bonus' link and request testimonial."
        else:
            return "Customer Success: Send 'Loyalty Appreciation' gift."
    return "No Specific Action Required. Monitor usage."

# ===============================
# 3. Load Churn Prediction Model
# ===============================
churn_model = None
try:
    churn_model = joblib.load("tuned_logistic_regression.pkl")
    st.success("✅ Churn Prediction Model loaded successfully!")
except Exception as e:
    st.warning(f"❌ Model not loaded: {e}. Predictions will not work.")

# ===============================
# 4. Streamlit Page Setup
# ===============================
st.set_page_config(page_title="Gym Member Retention Dashboard", layout="wide")
st.title("💪 AI-Powered Gym Member Insights")

# ===============================
# 5. Tabs
# ===============================
tab1, tab2 = st.tabs(["📉 Churn Prediction", "💬 Feedback Diagnostic"])

# -------------------------------
# Tab 1: Churn Prediction
# -------------------------------
with tab1:
    st.header("📉 Predict Churn Probability")
    customer_id = st.text_input("Customer ID", "GYM_0001")
    gender = st.selectbox("Gender", ["Male", "Female"])
    near_location = st.selectbox("Near Gym Location", ["Near", "Far"])
    partner = st.selectbox("Has Partner Membership", ["Yes", "No"])
    promo_friends = st.selectbox("Used Friend Promo", ["Yes", "No"])
    phone = st.selectbox("Has Active Phone Contact", ["Yes", "No"])
    contract_period = st.selectbox("Contract Period (months)", [6, 12])
    group_visits = st.selectbox("Attends Group Visits", ["Solo", "With friends"])
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    avg_additional_charges_total = st.number_input("Average Additional Charges", min_value=0.0, step=0.1)
    lifetime_in_gym = st.number_input("Lifetime in Gym (months)", min_value=0, step=1)
    avg_class_frequency_current_month = st.number_input("Avg Class Frequency (Current Month)", min_value=0.0, step=0.1)

    input_data = pd.DataFrame({
        "gender": [gender],
        "Near_Location": [near_location],
        "Partner": [partner],
        "Promo_friends": [promo_friends],
        "Phone": [phone],
        "Contract_period": [contract_period],
        "Group_visits": [group_visits],
        "Age": [age],
        "Avg_additional_charges_total": [avg_additional_charges_total],
        "Lifetime_in_the _gym": [lifetime_in_gym],
        "Avg_class_frequency_current_month": [avg_class_frequency_current_month]
    })

    if st.button("🔍 Predict Churn"):
        if churn_model:
            try:
                churn_pred = churn_model.predict(input_data)[0]
                churn_prob = churn_model.predict_proba(input_data)[0][1]

                if churn_pred == 1:
                    st.error(f"⚠️ Member {customer_id} likely to churn! (Probability: {churn_prob:.2f})")
                    st.session_state['churn_status'] = 'Left'
                else:
                    st.success(f"✅ Member {customer_id} likely to stay! (Probability: {churn_prob:.2f})")
                    st.session_state['churn_status'] = 'Stayed'

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Churn model not loaded.")

# -------------------------------
# Tab 2: Feedback Diagnostic
# -------------------------------
with tab2:
    st.header("💬 Diagnostic Engine & Recommendations")
    churn_status_default = st.session_state.get('churn_status', 'Left')
    churn_status_input = st.selectbox("Churn Status:", ["Left", "Stayed"], index=0 if churn_status_default=='Left' else 1)
    feedback = st.text_area("Customer Feedback:", 
                            "I decided to leave. Frankly, the additional charges were too high.", height=150)

    if st.button("🧠 Analyze Feedback"):
        if not feedback.strip():
            st.warning("Please provide feedback text.")
        else:
            processed_text = preprocess_text(feedback)
            topic_name, topic_id = determine_topic(processed_text)
            recommendation = generate_recommendation(topic_name, churn_status_input, feedback)

            st.markdown("---")
            st.subheader("✅ Analysis Output")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Churn Status Used", value=churn_status_input, delta=f"Topic ID: {topic_id}")
                st.markdown(f"**Sentiment Diagnosis:** {'🚨 High Risk/Negative' if topic_id >= 3 else '🟢 Positive/Low Risk'}")
            with col2:
                st.markdown("##### 🎯 Recommended Action:")
                if topic_id >= 3 and churn_status_input == 'Left':
                     st.error(recommendation)
                elif topic_id >= 3:
                     st.warning(recommendation)
                else:
                     st.success(recommendation)

st.markdown("---")
st.caption("🤖 Built with Streamlit | Gym Member Retention Dashboard")
