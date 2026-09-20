from sqlalchemy import label
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from env_var import *
from scraper import pull_reviews, clean_text, extract_asin
import time

# 1. Page Configuration & Setup
st.set_page_config(page_title="PULSAR", layout="wide", page_icon="🛸")
st.title("📊 PULSAR: Aspect-Based Sentiment Tracker")
st.markdown(unsafe_allow_html=True, body="<font size=3 color=cream>Analyze product reviews to uncover operational insights and sentiment trends across business aspects.</font>")
st.caption("Extract operational product insights from raw text reviews using NLP.")


# Securely prompt for the API key in the UI sidebar
with st.sidebar:
    st.header("🔑 API Configuration")
    hf_token = st.text_input("Hugging Face Access Token", type="password", help="Get a free token from huggingface.co/settings/tokens")
    st.markdown("---")
    st.markdown("### Interview Metrics Simulated")
    st.metric("Model Latency", "~1.2s", "-0.3s vs Local")
    st.metric("API Cost", "$0.00", "Using HF Free Tier")

st.subheader("📥 Input Product Reviews")
user_input = st.text_input(label="Paste the target Product URL from Amazon.in only", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None)

API_URL = zero_shot_model  # default; reassigned before each analysis stage
headers = {
    "Authorization": f"Bearer {hf_token}",
}

def query(payload, max_retries=4, wait_seconds=5):
    """Call the HF Inference API, retrying on transient failures
    (model loading / rate limits) and returning [] on real errors."""
    for attempt in range(1, max_retries + 1):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        # Transient conditions worth retrying
        if response.status_code in (503, 429, 500, 502):
            if attempt < max_retries:
                time.sleep(wait_seconds)
                continue
        # Real error: show what the server actually said
        st.warning(
            f"Hugging Face API error | Retrying... {response.status_code}: {response.text[:300]}"
        )
        return []
    st.error("Hugging Face API is busy/unavailable after several retries. Try again shortly.")
    return []

if st.button("🚀 Analyze Sentiment Pulse"):
    if not hf_token:
        st.error("Please provide a Hugging Face Access Token in the sidebar.")
    elif not user_input:
        st.warning("Please enter at least one user_input product url to analyze.")

    elif user_input:
        # Cache the scraped reviews in session state so we only pull
        # again when the URL actually changes
        if (
            "df_req2" not in st.session_state
            or st.session_state.get("req2_url") != user_input
        ):
            st.session_state["df_req2"] = pull_reviews(url_input=user_input)
            st.session_state["req2_url"] = user_input
        df_req2 = st.session_state["df_req2"]
        text = df_req2['cleaned_text'].unique()
        # 2. Aspect-Based Classification
        st.subheader("📊 Aspect-Based Classification is being processed...")
        API_URL = zero_shot_model
        output_list = []
        for item in text:
            output_class = query({
                "inputs": item,
                "parameters": {"candidate_labels": ["Looks","Product Quality", "Shipping & Delivery", "Customer Service", "Pricing & Value","Ratings",
                                                   "Satisfaction","Features & Functionality", "Returns & Refunds", "Website & App Experience"]},
            })
            output_list.append(output_class)
        flat_list = [item for sublist in output_list for item in sublist]
        df_all = pd.DataFrame(flat_list)
        df_average = (
            df_all.groupby('label', as_index=False)['score']
            .mean()
            .sort_values(by='score', ascending=False)
            .reset_index(drop=True)
        )
        df_average['score'] = df_average['score'] * 100
        df_average.rename(columns={'label': 'Operational Aspect', 'score': 'Frequency of people talking about the aspect (%)'}, inplace=True)
        df_average = df_average.round(2)
        st.dataframe(df_average)
        # sentiment analysis
        st.subheader("📊 Abstract Sentiment Analysis")
        API_URL = sentiment_model
        final_list = []
        for item in df_req2['cleaned_text'].unique():
            output = query({"inputs": item,})
            if output:
                final_list.append(output)
        flattened_data = [item[0] for item in final_list if item]
        pivoted_data = []
        for text_scores in flattened_data:
            row_dict = {d['label']: d['score'] for d in text_scores}
            pivoted_data.append(row_dict)
        df_pivoted = pd.DataFrame(pivoted_data)
        df_pivoted = df_pivoted.round(4)
        st.dataframe(df_pivoted)

        st.subheader("📊 Overall Sentiment Analysis")
        API_URL = sentiment_analyzer

        output_list = []
        for item in text:
            output_class = query({
                "inputs": item,
            })
            output_list.append(output_class)

        flattened_data = [item[0] for item in output_list if item]
        pivoted_data = []
        for text_scores in flattened_data:
            row_dict = {d['label']: d['score'] for d in text_scores}
            pivoted_data.append(row_dict)
        df_pivoted_overall = pd.DataFrame(pivoted_data)
        df_pivoted_overall = df_pivoted_overall.round(2)
        df_pivoted_overall['Negative'] = df_pivoted_overall['Negative'] + df_pivoted_overall['Very Negative']
        df_pivoted_overall['Positive'] = df_pivoted_overall['Positive'] + df_pivoted_overall['Very Positive']
        st.dataframe(df_pivoted_overall)
        positive_sentiment = round(df_pivoted_overall['Positive'].mean()*100,2)
        Negative_sentiment = round(df_pivoted_overall['Negative'].mean()*100,2)
        Neutral_sentiment = round(df_pivoted_overall['Neutral'].mean()*100,2)
        st.metric("Total Reviews Processed", len(df_req2))

        col1, col2, col3 = st.columns(3)

        col1.metric("Positive Sentiment for this product", positive_sentiment, "%", delta_color= "normal")
        col1.metric("Negative Sentiment for this product", Negative_sentiment, "%", delta_color= "inverse")
        col1.metric("Neutral Sentiment for this product", Neutral_sentiment, "%", delta_color= "off")
        most_talked = df_average['Operational Aspect'].iloc[0] if not df_average.empty else "None"
        second_most_talked = df_average['Operational Aspect'].iloc[1] if not df_average.empty else "None"
        third_most_talked = df_average['Operational Aspect'].iloc[2] if not df_average.empty else "None"
        col2.metric("Most talked about feature", most_talked)
        col2.metric("Second Most talked about feature", second_most_talked)
        col2.metric("Third Most talked about feature", third_most_talked)
        average_abstract_positive = (df_pivoted['joy'] + df_pivoted['love']).mean()
        average_abstract_negative = (df_pivoted['anger'] + df_pivoted['sadness'] + df_pivoted['fear']).mean()
        average_Abstract_sentiment = average_abstract_positive - average_abstract_negative
        col3.metric("Average Abstract Sentiment", round(average_Abstract_sentiment,2), delta_color="off" if average_Abstract_sentiment > 0 else "normal")
        col3.metric("Average Abstract positive Sentiment", round(average_abstract_positive,2), delta_color="normal")
        col3.metric("Average Abstract Negative Sentiment", round(average_abstract_negative,2), delta_color="inverse")

        # Charts Section
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.write("#### Sentiment Distribution")
            fig_pie = px.pie(df_average, names="Operational Aspect",values="Frequency of people talking about the aspect (%)", color="Operational Aspect")
            st.plotly_chart(fig_pie, width='stretch')
        with chart_col2:
            df_chart = df_pivoted_overall.mean().reset_index()
            df_chart.columns = ['Sentiment', 'Average Score']
            st.write("#### Complaints & Praise by Department")
            fig_bar = px.histogram(df_chart, x="Sentiment",y='Average Score', color="Average Score", barmode="group",
                                   color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral/Disappointed": "#f1c40f"})
            st.plotly_chart(fig_bar, width='stretch')
