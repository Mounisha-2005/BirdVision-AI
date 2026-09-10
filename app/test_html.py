import streamlit as st

st.set_page_config(page_title="HTML Test", layout="wide")

st.markdown("""
<div style="
    padding:40px;
    background:#111827;
    color:white;
    border-radius:20px;
    text-align:center;
">
    <h1>HTML TEST</h1>
    <div style="font-size:20px;">
        THIS SHOULD APPEAR AS A STYLED BOX
    </div>
</div>
""", unsafe_allow_html=True)