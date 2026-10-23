import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import accelerate

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scraper import pull_reviews, clean_text, extract_asin


# 1. Page Configuration & Setup
st.set_page_config(page_title="PULSAR", layout="wide", page_icon="🛸")

st.title("📊 PULSAR: Aspect-Based Sentiment Tracker")

st.markdown(
    unsafe_allow_html=True,
    body="<font size=3 color=cream>Analyze product reviews to uncover operational insights and sentiment trends across business aspects.</font>"
)

st.caption("Extract operational product insights from raw text reviews using NLP.")


# Local Model Configuration
with st.sidebar:
    st.header("🤖 Local Model Configuration")
    st.success("Models run locally using Transformers")

    st.markdown("---")
    st.markdown("### Interview Metrics Simulated")
    st.metric("Inference Mode", "Local")
    st.metric("API Cost", "$0.00", "No inference API calls")


st.subheader("📥 Input Product Reviews")

user_input = st.text_input(
    label="Paste the target Product URL from Amazon.in only",
    value="",
    help="Enter an Amazon.in product URL."
)


# 2. Load and cache the models
HF_TOKEN = st.secrets["hf_key"]
@st.cache_resource
def load_model(model_name):
    """Load a Hugging Face model and tokenizer once, then reuse them."""

    tokenizer = AutoTokenizer.from_pretrained(model_name,token=HF_TOKEN)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="auto",
        token=HF_TOKEN
    )

    model.eval()

    return tokenizer, model


# Read model repository IDs from Streamlit secrets
ZERO_SHOT_MODEL = st.secrets["zero_shot_model"]
ABSTRACT_SENTIMENT_MODEL = st.secrets["sentiment_model"]
OVERALL_SENTIMENT_MODEL = st.secrets["sentiment_analyzer"]


# 3. Local replacement for the Hugging Face Inference API
def query(payload, model_key="zero_shot_model"):

    try:
        # Select the model for the current analysis stage
        model_names = {
            "zero_shot_model": ZERO_SHOT_MODEL,
            "sentiment_model": ABSTRACT_SENTIMENT_MODEL,
            "sentiment_analyzer": OVERALL_SENTIMENT_MODEL
        }

        model_name = model_names[model_key]
        tokenizer, model = load_model(model_name)

        # Extract the same payload fields used by the API version
        item = payload["inputs"]
        parameters = payload.get("parameters", {})

        # Tokenize the input
        if "candidate_labels" in parameters:

            # -------------------------------------------
            # Zero-shot classification
            # -------------------------------------------
            candidate_labels = parameters["candidate_labels"]

            if not candidate_labels:
                raise ValueError("candidate_labels cannot be empty.")

            hypothesis_template = parameters.get(
                "hypothesis_template",
                "This example is {}."
            )

            multi_label = parameters.get("multi_label", False)

            hypotheses = [
                hypothesis_template.format(label)
                for label in candidate_labels
            ]

            inputs = tokenizer(
                [item] * len(candidate_labels),
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)

            logits = outputs.logits

            # Find the entailment and contradiction class indices
            label2id = {
                str(key).lower(): value
                for key, value in model.config.label2id.items()
            }

            entailment_id = next(
                (
                    value for label, value in label2id.items()
                    if "entailment" in label
                ),
                2
            )

            contradiction_id = next(
                (
                    value for label, value in label2id.items()
                    if "contradiction" in label
                ),
                0
            )

            if multi_label:
                # Score each candidate label independently
                relevant_logits = logits[
                    :, [contradiction_id, entailment_id]
                ]

                scores = torch.softmax(
                    relevant_logits, dim=1
                )[:, 1]

            else:
                # Normalize scores across candidate labels
                scores = torch.softmax(
                    logits[:, entailment_id], dim=0
                )

            scores = scores.cpu().tolist()

            results = [
                {"label": label, "score": score}
                for label, score in zip(candidate_labels, scores)
            ]

            results.sort(
                key=lambda result: result["score"],
                reverse=True
            )

            return results

        else:

            # -------------------------------------------
            # Standard classification: sentiment/emotion
            # -------------------------------------------
            inputs = tokenizer(
                item,
                return_tensors="pt",
                truncation=True
            )

            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)

            # Convert logits to class probabilities
            scores = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

            id2label = model.config.id2label

            results = [
                {
                    "label": id2label.get(i, str(i)),
                    "score": score
                }
                for i, score in enumerate(scores)
            ]

            results.sort(
                key=lambda result: result["score"],
                reverse=True
            )

            return results

    except Exception as e:
        st.error(f"Local model inference error: {e}")
        return []


# 4. Analyze Reviews
if st.button("🚀 Analyze Sentiment Pulse"):

    if not user_input:
        st.warning("Please enter a product URL to analyze.")

    else:

        # Cache scraped reviews for the same URL
        if (
            "df_req2" not in st.session_state
            or st.session_state.get("req2_url") != user_input
        ):
            st.session_state["df_req2"] = pull_reviews(
                url_input=user_input
            )
            st.session_state["req2_url"] = user_input

        df_req2 = st.session_state["df_req2"]

        if df_req2 is None or df_req2.empty:
            st.warning("No reviews were retrieved for this product.")

        else:

            text = (
                df_req2["cleaned_text"]
                .dropna()
                .astype(str)
                .unique()
            )

            if len(text) == 0:
                st.warning("No valid review text was found.")

            else:

                # -------------------------------------------
                # 5. Aspect-Based Classification
                # -------------------------------------------
                st.subheader(
                    "📊 Aspect-Based Classification is being processed..."
                )

                output_list = []

                for item in text:
                    output_class = query(
                        {
                            "inputs": item,
                            "parameters": {
                                "candidate_labels": [
                                    "Looks",
                                    "Product Quality",
                                    "Shipping & Delivery",
                                    "Customer Service",
                                    "Pricing & Value",
                                    "Ratings",
                                    "Satisfaction",
                                    "Features & Functionality",
                                    "Returns & Refunds",
                                    "Website & App Experience"
                                ]
                            }
                        },
                        model_key="zero_shot_model"
                    )

                    output_list.append(output_class)

                flat_list = [
                    item
                    for sublist in output_list
                    for item in sublist
                ]

                df_all = pd.DataFrame(flat_list)

                if not df_all.empty:
                    df_average = (
                        df_all.groupby("label", as_index=False)["score"]
                        .mean()
                        .sort_values(
                            by="score",
                            ascending=False
                        )
                        .reset_index(drop=True)
                    )

                    df_average["score"] *= 100

                    df_average.rename(
                        columns={
                            "label": "Operational Aspect",
                            "score": (
                                "Frequency of people talking "
                                "about the aspect (%)"
                            )
                        },
                        inplace=True
                    )

                    df_average = df_average.round(2)

                    st.dataframe(df_average, width='stretch')

                else:
                    df_average = pd.DataFrame(
                        columns=[
                            "Operational Aspect",
                            "Frequency of people talking about the aspect (%)"
                        ]
                    )
                    st.warning(
                        "Aspect classification did not return any results."
                    )

                # -------------------------------------------
                # 6. Abstract Sentiment / Emotion Analysis
                # -------------------------------------------
                st.subheader("📊 Abstract Sentiment Analysis")

                final_list = []

                for item in text:
                    output = query(
                        {"inputs": item},
                        model_key="sentiment_model"
                    )

                    if output:
                        final_list.append(output)

                flattened_data = [
                    item
                    for item in final_list
                    if item
                ]

                pivoted_data = []

                for text_scores in flattened_data:
                    row_dict = {
                        d["label"]: d["score"]
                        for d in text_scores
                    }
                    pivoted_data.append(row_dict)

                df_pivoted = pd.DataFrame(pivoted_data)
                df_pivoted = df_pivoted.round(4)

                st.dataframe(df_pivoted, width='stretch')

                # -------------------------------------------
                # 7. Overall Sentiment Analysis
                # -------------------------------------------
                st.subheader("📊 Overall Sentiment Analysis")

                output_list = []

                for item in text:
                    output_class = query(
                        {"inputs": item},
                        model_key="sentiment_analyzer"
                    )

                    output_list.append(output_class)

                flattened_data = [
                    item
                    for item in output_list
                    if item
                ]

                pivoted_data = []

                for text_scores in flattened_data:
                    row_dict = {
                        d["label"]: d["score"]
                        for d in text_scores
                    }
                    pivoted_data.append(row_dict)

                df_pivoted_overall = pd.DataFrame(pivoted_data)
                df_pivoted_overall = df_pivoted_overall.round(2)

                # Preserve your original sentiment aggregation
                if {
                    "Negative", "Very Negative",
                    "Positive", "Very Positive", "Neutral"
                }.issubset(df_pivoted_overall.columns):

                    df_pivoted_overall["Negative"] = (
                        df_pivoted_overall["Negative"]
                        + df_pivoted_overall["Very Negative"]
                    )

                    df_pivoted_overall["Positive"] = (
                        df_pivoted_overall["Positive"]
                        + df_pivoted_overall["Very Positive"]
                    )

                st.dataframe(
                    df_pivoted_overall,
                    width='stretch'
                )

                # -------------------------------------------
                # 8. Summary Metrics
                # -------------------------------------------
                required_sentiments = {
                    "Positive", "Negative", "Neutral"
                }

                if required_sentiments.issubset(
                    df_pivoted_overall.columns
                ) and not df_pivoted_overall.empty:

                    positive_sentiment = round(
                        df_pivoted_overall["Positive"].mean() * 100, 2
                    )

                    negative_sentiment = round(
                        df_pivoted_overall["Negative"].mean() * 100, 2
                    )

                    neutral_sentiment = round(
                        df_pivoted_overall["Neutral"].mean() * 100, 2
                    )

                else:
                    positive_sentiment = 0
                    negative_sentiment = 0
                    neutral_sentiment = 0

                st.metric("Total Reviews Processed", len(df_req2))

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Positive Sentiment for this product",
                    f"{positive_sentiment}%",
                    delta_color="normal"
                )

                col1.metric(
                    "Negative Sentiment for this product",
                    f"{negative_sentiment}%",
                    delta_color="inverse"
                )

                col1.metric(
                    "Neutral Sentiment for this product",
                    f"{neutral_sentiment}%",
                    delta_color="off"
                )

                most_talked = (
                    df_average["Operational Aspect"].iloc[0]
                    if not df_average.empty else "None"
                )

                second_most_talked = (
                    df_average["Operational Aspect"].iloc[1]
                    if len(df_average) > 1 else "None"
                )

                third_most_talked = (
                    df_average["Operational Aspect"].iloc[2]
                    if len(df_average) > 2 else "None"
                )

                col2.metric("Most talked about feature", most_talked)
                col2.metric(
                    "Second Most talked about feature",
                    second_most_talked
                )
                col2.metric(
                    "Third Most talked about feature",
                    third_most_talked
                )

                # Abstract sentiment metrics
                if {"joy", "love"}.issubset(df_pivoted.columns):
                    average_abstract_positive = (
                        df_pivoted["joy"] + df_pivoted["love"]
                    ).mean()
                else:
                    average_abstract_positive = 0

                if {"anger", "sadness", "fear"}.issubset(
                    df_pivoted.columns
                ):
                    average_abstract_negative = (
                        df_pivoted["anger"]
                        + df_pivoted["sadness"]
                        + df_pivoted["fear"]
                    ).mean()
                else:
                    average_abstract_negative = 0

                average_Abstract_sentiment = (
                    average_abstract_positive
                    - average_abstract_negative
                )

                col3.metric(
                    "Average Abstract Sentiment",
                    round(average_Abstract_sentiment, 2),
                    delta_color=(
                        "off"
                        if average_Abstract_sentiment > 0
                        else "normal"
                    )
                )

                col3.metric(
                    "Average Abstract positive Sentiment",
                    round(average_abstract_positive, 2),
                    delta_color="normal"
                )

                col3.metric(
                    "Average Abstract Negative Sentiment",
                    round(average_abstract_negative, 2),
                    delta_color="inverse"
                )

                # -------------------------------------------
                # 9. Charts Section
                # -------------------------------------------
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.write("#### Sentiment Distribution")

                    if not df_average.empty:
                        fig_pie = px.pie(
                            df_average,
                            names="Operational Aspect",
                            values=(
                                "Frequency of people talking "
                                "about the aspect (%)"
                            ),
                            color="Operational Aspect"
                        )

                        st.plotly_chart(
                            fig_pie,
                            width="stretch"
                        )

                with chart_col2:
                    if not df_pivoted_overall.empty:
                        df_chart = (
                            df_pivoted_overall.mean()
                            .reset_index()
                        )

                        df_chart.columns = [
                            "Sentiment",
                            "Average Score"
                        ]

                        st.write(
                            "#### Complaints & Praise by Department"
                        )

                        fig_bar = px.bar(
                            df_chart,
                            x="Sentiment",
                            y="Average Score",
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#2ecc71",
                                "Negative": "#e74c3c",
                                "Neutral": "#f1c40f"
                            }
                        )

                        st.plotly_chart(
                            fig_bar,
                            width="stretch"
                        )