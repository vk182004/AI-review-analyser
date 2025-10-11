import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------------------------------------------
# 🌟 STEP 3: Presentation Touches – Page Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="E-commerce Escalation Engine",
    page_icon="🛍",
    layout="wide"
)

# Sidebar branding & about section
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/891/891462.png", width=80)
st.sidebar.title("🛍 E-commerce Escalation Engine")
st.sidebar.markdown(
    "*By Team ⊥ʜᴇ  𝐔ꪀв๏ʀͶ
*  \n"
    "AI-powered customer feedback analysis for small businesses."
)
st.sidebar.markdown("---")

# Main heading
st.title("🛍 E-commerce Escalation Engine")
st.markdown("### AI-powered customer feedback analysis for small businesses")

# ---------------------------------------------------
# NEW FUNCTION: Email Notification System
# ---------------------------------------------------
def send_urgency_email(high_priority_issues, admin_email):
    """Send email notification for urgent issues"""
    try:
        # Email configuration (set these in Streamlit secrets)
        smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = st.secrets.get("SMTP_PORT", 587)
        sender_email = st.secrets.get("SENDER_EMAIL")
        sender_password = st.secrets.get("SENDER_PASSWORD")
        
        if not all([sender_email, sender_password, admin_email]):
            st.error("❌ Email configuration incomplete. Check Streamlit secrets.")
            return False

        # Create email content
        subject = f"🚨 URGENT: {len(high_priority_issues)} Critical Customer Issues Need Attention"
        
        # Build email body
        body = f"""
        <html>
        <body>
            <h2 style="color: #ff4b4b;">🚨 Urgent Customer Issues Detected</h2>
            <p><strong>Total Critical Issues:</strong> {len(high_priority_issues)}</p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Issues Requiring Immediate Attention:</h3>
        """
        
        for i, issue in enumerate(high_priority_issues.iterrows(), 1):
            _, row = issue
            body += f"""
            <div style="border-left: 4px solid #ff4b4b; padding: 10px; margin: 15px 0; background: #fff5f5;">
                <p><strong>Issue #{i}:</strong></p>
                <p><strong>Category:</strong> {row['category']}</p>
                <p><strong>Review:</strong> "{row['review_text']}"</p>
                <p><strong>Rating:</strong> {row['rating']}/5</p>
                <p><strong>Customer:</strong> {row['customer_name']}</p>
                <p><strong>Date:</strong> {row['date']}</p>
            </div>
            """
        
        body += """
            <p><em>Please log in to the E-commerce Escalation Engine to address these issues promptly.</em></p>
            <hr>
            <p style="color: #666; font-size: 12px;">This is an automated notification from your E-commerce Escalation Engine.</p>
        </body>
        </html>
        """
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = admin_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        st.sidebar.success(f"📧 Email sent to {admin_email}")
        return True
        
    except Exception as e:
        st.error(f"❌ Email notification failed: {str(e)}")
        return False

def check_and_notify_urgent_issues(final_df):
    """Check for urgent issues and send email notifications AUTOMATICALLY"""
    high_priority_issues = final_df[final_df['urgency'] == 'HIGH']
    
    if len(high_priority_issues) == 0:
        return
    
    # Show browser notification
    st.toast(f"🚨 {len(high_priority_issues)} urgent issues detected!", icon="🚨")
    
    # Check if email notifications are enabled and configured
    if (st.session_state.get('enable_email_notifications', False) and 
        st.session_state.get('admin_email', '')):
        
        admin_email = st.session_state.get('admin_email')
        
        # Send email notification AUTOMATICALLY
        with st.sidebar:
            with st.expander("📧 Automatic Alert Sent", expanded=True):
                st.write(f"🚨 {len(high_priority_issues)} urgent issues detected!")
                st.write(f"📧 Sent alert to: {admin_email}")
                if send_urgency_email(high_priority_issues, admin_email):
                    st.success("✅ Email notification sent successfully!")
                else:
                    st.error("❌ Failed to send email notification")

# ---------------------------------------------------
# NEW FUNCTION: Local AI Model Summary
# ---------------------------------------------------
def local_ai_summary(final_df):
    """Generate summary using local transformers model"""
    try:
        from transformers import pipeline
        
        # Load a summarization model (will download on first run)
        @st.cache_resource
        def load_summarizer():
            return pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",
                min_length=30,
                max_length=150
            )
        
        summarizer = load_summarizer()
        
        # Prepare text for summarization - combine reviews and statistics
        review_samples = ". ".join(final_df['review_text'].head(8).tolist())
        
        # Calculate key metrics
        total_reviews = len(final_df)
        positive_reviews = len(final_df[final_df['sentiment'] == "POSITIVE"])
        negative_reviews = len(final_df[final_df['sentiment'] == "NEGATIVE"])
        high_urgency = len(final_df[final_df['urgency'] == "HIGH"])
        avg_rating = final_df['rating'].mean()
        top_categories = final_df['category'].value_counts().head(2).index.tolist()
        
        # Create context for the AI
        context = f"""
        E-commerce customer feedback analysis:
        Total reviews: {total_reviews}. Positive: {positive_reviews}. Negative: {negative_reviews}. 
        High urgency issues: {high_urgency}. Average rating: {avg_rating:.1f}/5. 
        Main categories: {', '.join(top_categories)}.
        Sample reviews: {review_samples}
        """
        
        # Truncate if too long for the model
        if len(context) > 1024:
            context = context[:1024]
            
        # Generate summary
        summary = summarizer(
            context, 
            max_length=120, 
            min_length=60, 
            do_sample=False,
            truncation=True
        )
        
        return summary[0]['summary_text']
        
    except Exception as e:
        st.error(f"Local AI model error: {str(e)}")
        return None

# ---------------------------------------------------
# NEW FUNCTION: Enhanced Rule-Based Summary
# ---------------------------------------------------
def enhanced_rule_based_summary(final_df):
    """Generate intelligent summary without AI APIs"""
    
    # Calculate comprehensive metrics
    total_reviews = len(final_df)
    positive_reviews = len(final_df[final_df['sentiment'] == "POSITIVE"])
    negative_reviews = len(final_df[final_df['sentiment'] == "NEGATIVE"])
    high_urgency = len(final_df[final_df['urgency'] == "HIGH"])
    
    # Category analysis
    category_issues = final_df[final_df['sentiment'] == 'NEGATIVE']['category'].value_counts()
    top_issue_categories = category_issues.head(3).index.tolist()
    positive_categories = final_df[final_df['sentiment'] == 'POSITIVE']['category'].value_counts()
    top_positive_categories = positive_categories.head(2).index.tolist()
    
    # Rating analysis
    avg_rating = final_df['rating'].mean()
    low_ratings = len(final_df[final_df['rating'] <= 2])
    high_ratings = len(final_df[final_df['rating'] >= 4])
    
    # Sentiment strength
    positive_ratio = positive_reviews / total_reviews if total_reviews > 0 else 0
    
    # Generate insights based on data patterns
    if positive_ratio > 0.7:
        sentiment_insight = "excellent customer satisfaction"
        recommendation = "Continue maintaining high standards and consider collecting testimonials from happy customers."
    elif positive_ratio > 0.5:
        sentiment_insight = "generally positive feedback"
        recommendation = "Address minor issues to improve customer experience further and increase retention."
    else:
        sentiment_insight = "needs immediate attention"
        recommendation = "Priority review required for product/service quality improvements to prevent customer churn."
    
    # Urgency assessment
    if high_urgency > 5:
        urgency_status = "critical level"
        action = "🚨 Immediate action required on high-priority issues to prevent escalation."
    elif high_urgency > 2:
        urgency_status = "elevated level" 
        action = "⚠️ Monitor and address urgent issues promptly to maintain customer trust."
    else:
        urgency_status = "manageable level"
        action = "✅ Continue regular monitoring of customer feedback."
    
    # Generate comprehensive summary
    summary = f"""
📊 **AI-Powered Customer Feedback Analysis**

**Overall Performance:**
• Customer Sentiment: {sentiment_insight.title()} ({positive_reviews}/{total_reviews} positive reviews)
• Average Rating: {avg_rating:.1f}/5 stars ({high_ratings} high ratings, {low_ratings} low ratings)
• Urgent Issues: {high_urgency} cases ({urgency_status})

**Key Areas:**
• Top Concern Categories: {', '.join(top_issue_categories) if top_issue_categories else 'No major concerns'}
• Strong Performing Areas: {', '.join(top_positive_categories) if top_positive_categories else 'General positive feedback'}

**Action Plan:**
{recommendation}
{action}

**Business Insight:** {'🚨 Focus on quality and service improvements' if negative_reviews > positive_reviews else '✅ Strong customer satisfaction maintained across most categories'}
"""
    
    return summary

# ---------------------------------------------------
# NEW FUNCTION: Advanced Analytics
# ---------------------------------------------------
def calculate_advanced_metrics(df):
    """Calculate advanced business metrics"""
    total_reviews = len(df)
    if total_reviews == 0:
        return {}
    
    # Sentiment metrics
    positive_pct = (len(df[df['sentiment'] == 'POSITIVE']) / total_reviews) * 100
    negative_pct = (len(df[df['sentiment'] == 'NEGATIVE']) / total_reviews) * 100
    
    # Urgency metrics
    high_urgency_pct = (len(df[df['urgency'] == 'HIGH']) / total_reviews) * 100
    
    # Category distribution
    category_dist = df['category'].value_counts().to_dict()
    
    # Rating analysis
    avg_rating = df['rating'].mean()
    low_ratings = len(df[df['rating'] <= 2])
    
    return {
        'positive_percentage': round(positive_pct, 1),
        'negative_percentage': round(negative_pct, 1),
        'high_urgency_percentage': round(high_urgency_pct, 1),
        'category_distribution': category_dist,
        'average_rating': round(avg_rating, 1),
        'low_rating_count': low_ratings,
        'customer_satisfaction_score': max(0, round((avg_rating / 5) * 100, 1))
    }

# ---------------------------------------------------
# NEW FUNCTION: Priority Recommendations
# ---------------------------------------------------
def generate_recommendations(df):
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    high_priority_count = len(df[df['urgency'] == 'HIGH'])
    negative_count = len(df[df['sentiment'] == 'NEGATIVE'])
    
    # Recommendation logic
    if high_priority_count > 0:
        top_urgent_category = df[df['urgency'] == 'HIGH']['category'].mode()
        if len(top_urgent_category) > 0:
            recommendations.append(f"🚨 *Immediate Action Needed*: Address {high_priority_count} high-urgency issues in {top_urgent_category[0]} category")
    
    if negative_count > 5:
        recommendations.append("📊 *Deep Dive Required*: High volume of negative feedback. Consider product/service audit")
    
    # Category-specific recommendations
    category_issues = df[df['sentiment'] == 'NEGATIVE']['category'].value_counts()
    for category, count in category_issues.head(2).items():
        if count >= 3:
            recommendations.append(f"🔧 *Focus Area*: Improve {category} processes ({count} issues reported)")
    
    # Positive reinforcement
    positive_count = len(df[df['sentiment'] == 'POSITIVE'])
    if positive_count > 10:
        recommendations.append("⭐ *Strength*: Strong positive feedback indicates good customer experience in several areas")
    
    if not recommendations:
        recommendations.append("✅ *All Good*: Current metrics indicate healthy customer satisfaction levels")
    
    return recommendations

# ---------------------------------------------------
# NEW FUNCTION: Trend Analysis (Simulated)
# ---------------------------------------------------
def analyze_trends(df):
    """Analyze trends based on actual data patterns"""
    if len(df) == 0:
        return {
            'sentiment_trend': 'stable',
            'urgent_issues_trend': 'stable', 
            'top_category': 'N/A',
            'improvement_areas': []
        }
    
    # Calculate sentiment trend based on actual data
    positive_count = len(df[df['sentiment'] == 'POSITIVE'])
    negative_count = len(df[df['sentiment'] == 'NEGATIVE'])
    total_count = len(df)
    
    if negative_count > positive_count + (total_count * 0.2):  # If significantly more negative
        sentiment_trend = 'falling'
    elif positive_count > negative_count + (total_count * 0.2):  # If significantly more positive
        sentiment_trend = 'rising'
    else:
        sentiment_trend = 'stable'
    
    # Calculate urgency trend based on high urgency percentage
    high_urgency_count = len(df[df['urgency'] == 'HIGH'])
    urgency_percentage = (high_urgency_count / total_count) * 100
    
    if urgency_percentage > 30:  # More than 30% high urgency
        urgency_trend = 'increasing'
    elif urgency_percentage < 10:  # Less than 10% high urgency
        urgency_trend = 'decreasing'
    else:
        urgency_trend = 'stable'
    
    # Get actual top category from data
    top_category = df['category'].mode()[0] if len(df['category'].mode()) > 0 else 'general feedback'
    
    # Get improvement areas from negative reviews
    negative_categories = df[df['sentiment'] == 'NEGATIVE']['category'].value_counts()
    improvement_areas = list(negative_categories.head(2).index) if len(negative_categories) > 0 else []
    
    return {
        'sentiment_trend': sentiment_trend,
        'urgent_issues_trend': urgency_trend,
        'top_category': top_category,
        'improvement_areas': improvement_areas
    }

# ---------------------------------------------------
# Your Existing Code (untouched)
# ---------------------------------------------------

def analyze_review(text):
    text_lower = str(text).lower()
    positive_words = ['great', 'excellent', 'amazing', 'good', 'happy', 'perfect', 'love', 'fast', 'awesome', 'satisfied']
    negative_words = ['terrible', 'bad', 'poor', 'broken', 'wrong', 'disappointed', 'horrible', 'awful', 'never', 'worst', 'waste']
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if negative_count > positive_count:
        sentiment = "NEGATIVE"
    elif positive_count > negative_count:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEUTRAL"

    if any(word in text_lower for word in ['shipping', 'delivery', 'arrived', 'package', 'delivered']):
        category = "shipping/delivery"
    elif any(word in text_lower for word in ['quality', 'material', 'broke', 'broken', 'working', 'fell apart', 'durable']):
        category = "product quality"
    elif any(word in text_lower for word in ['service', 'support', 'help', 'rude', 'unhelpful', 'representative']):
        category = "customer service"
    elif any(word in text_lower for word in ['price', 'expensive', 'cost', 'money', 'worth', 'value']):
        category = "pricing"
    else:
        category = "general feedback"

    urgent_words = ['angry', 'refund', 'terrible', 'never again', 'worst', 'horrible', 'broken', 'wrong item', 'scam']
    urgency = "HIGH" if any(word in text_lower for word in urgent_words) else "MEDIUM" if sentiment == "NEGATIVE" else "LOW"

    return {'sentiment': sentiment, 'category': category, 'urgency': urgency}


def generate_mock_data():
    reviews = [
        "Great product! Fast shipping and excellent quality. Will buy again!",
        "Terrible experience. Product arrived broken and customer service was unhelpful.",
        "Slow shipping but the product itself is good quality for the price.",
        "Item never arrived. Very disappointed with this purchase.",
        "Excellent customer service! They helped me with my issue immediately.",
        "Poor quality material. Fell apart after one week of use.",
        "Fast delivery and the product looks exactly like the pictures. Happy with my purchase!",
        "Wrong item was sent. Now I have to go through the hassle of returning it.",
        "Price was too high for the quality received. Not worth it.",
        "Amazing product! Better than expected and arrived early.",
        "Package was damaged during shipping. The product inside was scratched.",
        "Customer service representative was rude and didn't solve my problem.",
        "Good value for money. Shipping took longer than expected though.",
        "Product stopped working after 2 days. Requesting a refund.",
        "Beautiful design and fast shipping. Very satisfied!",
    ]
    data = []
    for i, review in enumerate(reviews):
        data.append({
            'review_id': i + 1,
            'customer_name': f'Customer {i+1}',
            'review_text': review,
            'rating': random.randint(1, 5),
            'date': (datetime.now() - pd.Timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
        })
    return pd.DataFrame(data)

# Sidebar controls
st.sidebar.header("⚙ Controls")
use_mock_data = st.sidebar.checkbox("Use Demo Data", value=True)

if not use_mock_data:
    uploaded_file = st.sidebar.file_uploader("Upload customer reviews CSV", type=['csv'])
else:
    uploaded_file = None

st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Options")

# AI options
ai_option = st.sidebar.radio(
    "Choose Summary Method:",
    ["Enhanced Rule-Based", "Local AI Model"],
    help="Enhanced: Instant analysis, Local AI: True AI summary (slower first time)"
)

# DATA PROCESSING SECTION
if use_mock_data:
    df = generate_mock_data()
    st.sidebar.success("✅ Using demo data")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully")
    column_mapping = {'customer': 'customer_name','user': 'customer_name','name': 'customer_name',
                      'review': 'review_text','text': 'review_text','message': 'review_text',
                      'stars': 'rating','score': 'rating'}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
    if 'customer_name' not in df.columns:
        df['customer_name'] = [f'Customer {i+1}' for i in range(len(df))]
    if 'rating' not in df.columns:
        df['rating'] = 3
    if 'date' not in df.columns:
        df['date'] = datetime.now().strftime('%Y-%m-%d')
else:
    st.info("👈 Please upload a CSV file or use demo data to get started")
    st.stop()

with st.expander("📋 View Raw Data"):
    st.dataframe(df)

st.sidebar.info("🔄 Analyzing reviews...")
analysis_results = []
for _, row in df.iterrows():
    analysis = analyze_review(row['review_text'])
    analysis_results.append(analysis)

analysis_df = pd.DataFrame(analysis_results)
final_df = pd.concat([df, analysis_df], axis=1)

# ---------------------------------------------------
# NEW: Email Notification Settings - AFTER final_df is defined
# ---------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📧 Email Alerts")

enable_email_notifications = st.sidebar.checkbox(
    "Enable Email Notifications for Urgent Issues",
    value=st.session_state.get('enable_email_notifications', False),
    key='enable_email_notifications'
)

if enable_email_notifications:
    admin_email = st.sidebar.text_input(
        "Admin Email Address",
        value=st.session_state.get('admin_email', ''),
        placeholder="admin@company.com",
        help="Email where urgent issue alerts will be sent",
        key='admin_email'
    )
    
    # Test notification button
    if st.sidebar.button("🔔 Test Email Notification"):
        test_issues = final_df[final_df['urgency'] == 'HIGH'].head(1)
        if len(test_issues) > 0 and admin_email:
            if send_urgency_email(test_issues, admin_email):
                st.sidebar.success("Test email sent successfully!")
            else:
                st.sidebar.error("Failed to send test email")
        else:
            st.sidebar.warning("No urgent issues found for testing")

# ---------------------------------------------------
# NEW: Automatic Urgency Detection & Notification
# ---------------------------------------------------
check_and_notify_urgent_issues(final_df)

# ---------------------------------------------------
# NEW: Advanced Analytics Integration
# ---------------------------------------------------
st.header("📊 Performance Dashboard")

# Calculate advanced metrics
advanced_metrics = calculate_advanced_metrics(final_df)

col1, col2, col3, col4 = st.columns(4)
total_reviews = len(final_df)
negative_reviews = len(final_df[final_df['sentiment'] == 'NEGATIVE'])
high_urgency = len(final_df[final_df['urgency'] == 'HIGH'])
avg_rating = final_df['rating'].mean()

col1.metric("Total Reviews", total_reviews)
col2.metric("Negative Reviews", negative_reviews, 
           delta=f"{advanced_metrics.get('negative_percentage', 0)}%" if advanced_metrics else None)
col3.metric("Urgent Issues", high_urgency,
           delta=f"{advanced_metrics.get('high_urgency_percentage', 0)}%" if advanced_metrics else None)
col4.metric("Average Rating", f"{avg_rating:.1f}/5",
           delta=f"CSAT: {advanced_metrics.get('customer_satisfaction_score', 0)}%" if advanced_metrics else None)

# ---------------------------------------------------
# NEW: Recommendations Section
# ---------------------------------------------------
st.header("💡 Actionable Recommendations")
recommendations = generate_recommendations(final_df)

for i, recommendation in enumerate(recommendations):
    st.write(f"{recommendation}")

# ---------------------------------------------------
# NEW: Trends Analysis Section
# ---------------------------------------------------
st.header("📈 Trend Insights")
trends = analyze_trends(final_df)

col1, col2, col3 = st.columns(3)

with col1:
    trend_icon = "↗" if trends['sentiment_trend'] == 'rising' else "↘" if trends['sentiment_trend'] == 'falling' else "➡"
    st.metric("Sentiment Trend", trends['sentiment_trend'].title(), delta=trend_icon)

with col2:
    urgency_icon = "↘" if trends['urgent_issues_trend'] == 'decreasing' else "↗" if trends['urgent_issues_trend'] == 'increasing' else "➡"
    st.metric("Urgency Trend", trends['urgent_issues_trend'].title(), delta=urgency_icon)

with col3:
    st.metric("Top Category", trends['top_category'])

if trends['improvement_areas']:
    st.info(f"*Focus Areas*: {', '.join(trends['improvement_areas'])}")

# ---------------------------------------------------
# Existing Visualizations (untouched)
# ---------------------------------------------------
st.header("📈 Customer Insights")
col1, col2 = st.columns(2)
with col1:
    sentiment_counts = final_df['sentiment'].value_counts()
    fig_sentiment = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Customer Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={'POSITIVE': '#00D100', 'NEGATIVE': '#FF4B4B', 'NEUTRAL': '#FFC300'}
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
with col2:
    category_counts = final_df['category'].value_counts()
    fig_categories = px.bar(
        x=category_counts.values, y=category_counts.index,
        title="Top Issue Categories", orientation='h',
        labels={'x': 'Number of Reviews', 'y': 'Category'},
        color=category_counts.values
    )
    st.plotly_chart(fig_categories, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    urgency_counts = final_df['urgency'].value_counts()
    fig_urgency = px.bar(
        x=urgency_counts.index, y=urgency_counts.values,
        title="Issue Urgency Levels",
        color=urgency_counts.index,
        color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
    )
    st.plotly_chart(fig_urgency, use_container_width=True)
with col2:
    rating_counts = final_df['rating'].value_counts().sort_index()
    fig_ratings = px.bar(
        x=rating_counts.index, y=rating_counts.values,
        title="Customer Rating Distribution",
        labels={'x': 'Rating (1-5 stars)', 'y': 'Number of Reviews'}
    )
    st.plotly_chart(fig_ratings, use_container_width=True)

st.header("🚨 Critical Issues Requiring Attention")
high_priority_issues = final_df[final_df['urgency'] == 'HIGH']
if not high_priority_issues.empty:
    # Check if email was sent
    email_sent = (st.session_state.get('enable_email_notifications', False) and 
                  st.session_state.get('admin_email', ''))
    
    if email_sent:
        st.warning(f"**🚨 {len(high_priority_issues)} URGENT ISSUES DETECTED** - Email notification has been sent automatically!")
    else:
        st.warning(f"**🚨 {len(high_priority_issues)} URGENT ISSUES DETECTED** - Enable email alerts in sidebar to get notifications")
    
    for _, issue in high_priority_issues.iterrows():
        with st.container():
            st.error(f"""
            Category: {issue['category']}  
            Review: {issue['review_text']}  
            Rating: {issue['rating']}/5 • Customer: {issue['customer_name']}  
            Date: {issue['date']}
            """)
else:
    st.success("🎉 No critical issues detected! Customer satisfaction is good.")

st.header("📝 Detailed Review Analysis")
for _, review in final_df.iterrows():
    urgency_icon = "🔴" if review['urgency'] == 'HIGH' else "🟡" if review['urgency'] == 'MEDIUM' else "🟢"
    with st.expander(f"{urgency_icon} {review['review_text'][:80]}..."):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Sentiment: {review['sentiment']}")
            st.write(f"Category: {review['category']}")
        with col2:
            st.write(f"Urgency: {review['urgency']}")
            st.write(f"Rating: {review['rating']}/5")
            st.write(f"Customer: {review['customer_name']}")

# ---------------------------------------------------
# UPDATED: AI Summary Section with Local Model Only
# ---------------------------------------------------
st.header("🧠 AI Summary Report")

if ai_option == "Local AI Model":
    with st.spinner("🔄 Loading local AI model (first time may take 1-2 minutes to download)..."):
        ai_summary = local_ai_summary(final_df)
    
    if ai_summary:
        st.info(f"**🤖 Local AI Summary:**\n\n{ai_summary}")
        
        # Add retry button for different summary
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Generate New Summary"):
                st.rerun()
    else:
        st.warning("""
        **Local AI model is still loading or encountered an error.**
        
        This usually happens because:
        - First time setup requires downloading AI models (~1.5GB)
        - Limited memory on your system
        - The model is still initializing
        
        **What to do:**
        1. Wait a bit longer for the initial download
        2. Ensure you have: `pip install transformers torch`
        3. Try the 'Enhanced Rule-Based' option for instant results
        4. Refresh the page and try again
        
        Using enhanced rule-based summary for now.
        """)
        st.success(enhanced_rule_based_summary(final_df))

else:  # Enhanced Rule-Based
    st.success(enhanced_rule_based_summary(final_df))

# ---------------------------------------------------
# Export
# ---------------------------------------------------
st.sidebar.header("💾 Export Results")
csv_data = final_df.to_csv(index=False)
st.sidebar.download_button(
    "📥 Download Analysis Report",
    csv_data,
    file_name=f"escalation_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# Add email setup instructions
with st.sidebar.expander("🔧 Email Setup Guide"):
    st.markdown("""
    **Email Configuration:**
    
    1. **For Gmail:**
       - Enable 2-factor authentication
       - Generate an "App Password"
       - Use app password in SENDER_PASSWORD
    
    2. **Add to Streamlit secrets (.streamlit/secrets.toml):**
    ```toml
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SENDER_EMAIL = "your-email@gmail.com"
    SENDER_PASSWORD = "your-app-password"
    ```
    
    3. **Features:**
       - Automatic alerts for urgent issues
       - HTML formatted emails
       - Test button to verify setup
       - Real-time notifications
    """)

st.sidebar.success("🎯 Ready for demo!")
