import streamlit as st
import plotly.express as px
import pickle
import sys
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import load_assets
from utils.preprocessing import cleaned_text

st.set_page_config(page_title="📩 Spam Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>📩 Spam or Ham Message Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Made by: <b>Muhammad Abid Baihaqi Al Faridzi</b></p>", unsafe_allow_html=True)
st.markdown("---")

model, tokenizer = load_assets()

max_length = 100

def predict(text):
    cleaned = cleaned_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction_prob = model.predict(padded)[0][0]
    all_labels = {0: 'ham', 1: 'spam'}
    pred_idx = 1 if prediction_prob >= 0.5 else 0
    pred_label = all_labels[pred_idx]
    other_labels = all_labels[1-pred_idx]
    confidence_scores = prediction_prob if pred_idx == 1 else 1 - prediction_prob
    return pred_label, confidence_scores, other_labels

tab1, tab2 = st.tabs(["📥 Input Sentence", "📄 Input File"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Input Sentence")
        user_input = st.text_area("Message:", height=150, placeholder="Type your message here...")
        submitted = st.button("Predict")
        if submitted:
            if user_input.strip() == "":
                st.warning("Please enter a message to predict.")
            else:
                st.success("Check the prediction result on the right column.")
    with col2:
        st.markdown("### Prediction Result")
        if submitted and user_input.strip() != "":
            label, confidence, other_labels = predict(user_input)
            if label == 'ham':
                st.markdown(f"""
                            <div style='background: green; border-left: 5px solid #28a745; padding: 0.5em;'>
                            ✅ This message is classified as <b>{label.upper()}</b>
                            </div>""", unsafe_allow_html=True)
                label_color = '#28a745'
            else:
                st.markdown(f"""
                            <div style='background: red; border-left: 5px solid #dc3545; padding: 0.5em;'>
                            ❌ This message is classified as <b>{label.upper()}</b>
                            </div>""", unsafe_allow_html=True)
                label_color = '#dc3545'
                
            fig_data = pd.DataFrame({
                'Label': [label, other_labels],
                'Confidence': [confidence, 1-confidence]
            })

            fig = px.pie(
                fig_data,
                values='Confidence',
                names='Label',
                hole=0.6,
                color='Label',
                color_discrete_map={label: label_color, other_labels: '#dc3545' if label_color == '#28a745' else '#28a745'}
            )
            fig.update_traces(textinfo='none', hoverinfo='label+percent')
            fig.update_layout(
                showlegend=True,
                legend_title="Labels",
                annotations=[dict(text=f'{confidence*100:.2f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                title_text="Confidence Score", title_x=0.5
            )
            st.plotly_chart(fig, width="stretch")

with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('### Upload File')
        with st.form('csv_form'):
            uploaded_file = st.file_uploader("Upload a CSV file with a 'Pesan' column", type=['csv'])
            submit_file = st.form_submit_button("Predict File")
    with col2:
        st.markdown("### Prediction Results")
        if submit_file:
            if uploaded_file is None:
                st.warning("Please upload a CSV file to predict.")
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'Pesan' not in df.columns:
                        st.error("The uploaded CSV file must contain a 'Pesan' column.")
                    else:
                        df['cleaned_message'] = df['Pesan'].apply(cleaned_text)
                        sequences = tokenizer.texts_to_sequences(df['cleaned_message'])
                        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
                        predictions = model.predict(padded_sequences)
                        predicted_labels = []
                        confidence_scores = []
                        for pred in predictions:
                            pred_idx = 1 if pred[0] >= 0.5 else 0
                            all_labels = {0: 'ham', 1: 'spam'}
                            predicted_labels.append(all_labels[pred_idx])
                            conf = pred[0] if pred_idx == 1 else 1 - pred[0]
                            confidence_scores.append(conf)
                        df['predicted_label'] = predicted_labels
                        df['confidence_score'] = confidence_scores
                        st.success("Prediction completed. Here are the results:")
                        st.dataframe(df[['Pesan', 'predicted_label', 'confidence_score']])

                        csv_download = df[['Pesan', 'predicted_label', 'confidence_score']].to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions as CSV", data=csv_download, file_name="predictions.csv", mime='text/csv')
                
                    label_counts = df['predicted_label'].value_counts().reset_index()
                    fig = px.bar(
                        label_counts,
                        x='predicted_label',
                        y='count',
                        labels={'predicted_label': 'Label', 'count': 'Count'},
                        color='predicted_label',
                        color_discrete_map={'ham': '#28a745', 'spam': '#dc3545'},
                        title='Distribution of Spam and Ham Messages'
                    )
                    fig.update_traces(texttemplate='%{y}', textposition='outside')
                    fig.update_layout(yaxis=dict(title='Count'), xaxis=dict(title='Label'))
                    st.plotly_chart(fig, width="stretch")

                except Exception as e:
                    st.error(f"An error occurred while processing the file: {e}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:14px;'>© 2025 Muhammad Abid Baihaqi Al Faridzi</p>", unsafe_allow_html=True)






    






