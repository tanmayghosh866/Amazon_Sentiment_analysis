import requests
from apify_client import ApifyClient
import os
import pandas as pd
import re
import streamlit as st


def clean_text(text):
    text = ' '.join(text.strip().split())
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text



def extract_asin(product_url: str) -> str:
    patterns = [
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product-reviews/([A-Z0-9]{10})",
        r"[?&]asin=([A-Z0-9]{10})",
    ]
    for pattern in patterns:
        match = re.search(pattern, product_url, re.IGNORECASE)
        if match:
            return match.group(1).upper()
 
    raise ValueError(f"Could not find an ASIN in the URL: {product_url}")

def pull_reviews(url_input):
    prod_url = url_input
    st.write(f"🔗 Product URL: {prod_url}")
    ASIN = extract_asin(prod_url)
    st.write(ASIN)

    client = ApifyClient({st.secrets['apify_key']})

    # Prepare the Actor input
    run_input = {
        "productUrls": [{ "url": f"https://www.amazon.in/dp/{ASIN}" }],
        "maxReviews": 30,
        "includeGdprSensitive": False,
        "filterByRatings": ["fourStar", "oneStar"],
        "reviewsUseProductVariantFilter": False,
        "scrapeProductDetails": False,
        "reviewsAlwaysSaveCategoryData": False,
        "deduplicateRedirectedAsins": True,
    }
    run = client.actor("junglee/amazon-reviews-scraper").call(run_input=run_input)
    st.write(f"🕸️ Running query! spidey crawling reviews! 🕷️")
    
    dataset_items = list(client.dataset(run.default_dataset_id).iterate_items())
    df = pd.DataFrame(dataset_items)
    st.write("\n--- DataFrame Summary ---")
    st.write(f"Shape: {df.shape}")
    st.write("\n--- Available Columns ---")
    st.write(df.columns.tolist())
    df_req = df[['reviewTitle',
                'reviewDescription',
                'ratingScore',
                'date',
                'reviewReaction'
                ]]
    

    df_req2 = df_req.copy()
    df_req2 = df_req2.dropna(subset=["reviewDescription"])
    cleaned_text = []
    for item in df_req2['reviewDescription'].unique():
        clean_item = clean_text(item)
        cleaned_text.append(clean_item)
    df_req2['cleaned_text'] = cleaned_text
    df_req2.drop(columns=['reviewDescription'], inplace=True)
    return df_req2