# AI-Powered Gym Member Retention and Engagement System

## Abstract
This project presents an intelligent system that combines **machine learning (ML)** and **natural language processing (NLP)** to enhance gym member retention and engagement.  
The system predicts the likelihood of gym members discontinuing their memberships (**churn**) and analyzes member feedback using NLP for **sentiment, topic identification, and personalized recommendations**.  
By integrating predictive analytics with an AI feedback diagnostic engine, the project demonstrates a **data-driven approach to improving member retention and satisfaction** in the fitness industry.

---

## 1. Introduction
Member retention is a key challenge for fitness centers, as operational profitability depends heavily on long-term engagement. Traditional retention strategies often rely on manual follow-ups or static marketing campaigns, which can be inefficient.  

This project addresses this gap by developing an AI-powered **dashboard and feedback diagnostic system** that:

- Automatically predicts member churn.
- Analyzes member feedback for sentiment and engagement signals.
- Generates **actionable recommendations** for gym staff.
- Provides insights to **proactively engage members at risk of leaving**.

---

## 2. Problem Statement
Gym facilities often experience high churn due to inconsistent attendance, lack of motivation, or dissatisfaction. Existing approaches do not leverage data analytics or AI effectively to detect early warning signs.  

This project aims to:

- Predict gym member churn using historical and behavioral data.
- Analyze textual member feedback for sentiment and engagement topics.
- Provide actionable recommendations based on churn risk and feedback.
- Integrate a dashboard for gym owners to monitor retention metrics in real time.

---

## 3. Objectives
1. Build a **supervised machine learning model** for churn prediction (Logistic Regression used in the current implementation).  
2. Perform **sentiment analysis** on member feedback using a pre-trained ML model.  
3. Identify key **feedback topics** (via rule-based NLP logic) and provide actionable **recommendations**.  
4. Develop a **Streamlit dashboard** to visualize churn predictions, feedback sentiment, topics, and recommendations.  

---

## 4. Methodology

### 4.1 Data Collection and Preprocessing
The system uses a **Gym Customer Features and Churn Dataset** with:

- Demographics: age, gender, income
- Gym usage patterns: attendance frequency, membership duration
- Behavioral indicators: payment method, last visit date
- Target variable: **Churn** (Stayed / Left)

Textual feedback is preprocessed with:

- Lowercasing, punctuation removal, and tokenization
- Stopword removal
- Optional lemmatization (for topic analysis)

---

### 4.2 Machine Learning Model for Churn Prediction
The **Logistic Regression model** predicts the probability of a member leaving the gym.  

- Model evaluation metrics: Accuracy, Precision, Recall, F1-Score  
- Model stored as a **`.pkl` file** for deployment in the dashboard  
- Input features include demographic, behavioral, and membership-related attributes  

---

### 4.3 NLP for Sentiment Analysis and Feedback Diagnostics
The **Feedback Diagnostic Tab** uses a pre-trained sentiment analysis model (`sentiment_analysis_model.pkl`) to evaluate member feedback.  

- Feedback is **preprocessed** for tokenization and stopword removal.
- The model outputs **sentiment**: Positive / Negative.  
- A **topic-detection logic** identifies key themes in feedback such as:
  - Value & Quality
  - Hard Churn Signal
  - Usage / Integration Issues
  - Ambassadors / High Engagement
- Based on sentiment and topic, the system generates **actionable recommendations** for staff.

---

### 4.4 System Integration
The project consists of **three main components**:

1. **Data & Model Layer**: Preprocessing, churn prediction, sentiment analysis, topic classification.  
2. **Application Layer**: Streamlit dashboard for gym staff to monitor churn risk and member feedback.  
3. **Engagement Layer**: Recommendations engine that guides staff in engaging members at risk.

---

## 5. Results and Analysis
- The **churn prediction model** achieves high accuracy in identifying members likely to leave.  
- **Sentiment analysis** reliably identifies positive and negative feedback.  
- **Topic-based recommendations** allow staff to take **proactive retention actions**.  
- Streamlit dashboard visualizations provide an **intuitive interface** for monitoring churn, feedback sentiment, and engagement actions.

---

## 6. Expected Impact
- **Gym Owners**: Receive data-driven insights and actionable recommendations to improve retention.  
- **Gym Members**: Benefit from personalized and timely engagement.  
- **Data Science Practice**: Demonstrates a real-world integration of ML, NLP, and dashboard deployment.

---

## 7. Tools and Technologies

| Category | Tools / Libraries |
|----------|-----------------|
| Data Processing | Pandas, NumPy |
| Machine Learning | Logistic Regression (Scikit-learn), Random Forest, XGBoost |
| NLP | Scikit-learn, NLTK |
| Dashboard | Streamlit |
| Deployment | Streamlit, Joblib for model persistence |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git, GitHub |

---

## 8. Conclusion
This project demonstrates a **complete AI-powered system** for gym member retention and engagement. By combining **churn prediction**, **sentiment analysis**, and **feedback topic recommendations**, the system:

- Identifies at-risk members
- Provides actionable insights for staff
- Supports a **data-driven engagement strategy**

The framework can be extended to other domains where customer engagement and churn management are important.

---

## 9. Future Work
- Integrate **wearable fitness tracker data** for richer behavioral analytics.  
- Enhance recommendation logic with **reinforcement learning** for personalized engagement.  
- Support **voice-enabled chatbot interactions**.  
- Implement **multi-language NLP support** for international gyms.



## System Workflow

### How it works:
- **Gym Member Data** → cleaned & preprocessed for modeling.  
- **Churn Prediction Model** → identifies at-risk members.  
- **Feedback Diagnostic & Sentiment Analysis** → analyzes member feedback for sentiment.  
- **Topic Identification & Recommendations** → generates actionable steps for staff.  
- **Dashboard** → displays churn probabilities, sentiment, topics, and recommended actions in real time.



