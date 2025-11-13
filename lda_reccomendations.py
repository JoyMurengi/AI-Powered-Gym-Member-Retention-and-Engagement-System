# lda_recommendation.py
import re
import string

# -----------------------------
# 1. Text Preprocessing (for LDA)
# -----------------------------
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
    """Clean text for LDA topic assignment."""
    text = str(text).lower()
    text = text.replace("-", " ")  # keep phrases like "29-year-old"
    text = re.sub(f"[{re.escape(string.punctuation.replace('+',''))}]", "", text)
    words = re.findall(r'\b[a-z0-9]+\b', text)
    processed_words = [word for word in words if word not in COMMON_STOP_WORDS]
    return " ".join(processed_words)

# -----------------------------
# 2. Topic Assignment (Simple LDA-like engine)
# -----------------------------
def determine_topic(processed_text):
    """Assign topic ID and name based on keywords."""
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

# -----------------------------
# 3. Recommendation Logic
# -----------------------------
def generate_recommendation(topic_name, churn_status, raw_feedback):
    """Provide actionable recommendations based on topic + churn status."""
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
