import streamlit as st

st.set_page_config(page_title="LUE Portfolio Explorer", page_icon="📚", layout="wide")
st.title("📚 LUE Portfolio Explorer")

# Jump to dashboards (adjust names to match your files)
st.page_link("pages/1_🏭_Lab_View.py", label="Open Lab View", icon="🏭")
st.page_link("pages/2_🔬_Topic_View.py", label="Open Topic View (placeholder)", icon="🔬")
st.page_link("pages/3_👥_Author_View.py", label="Open Author View (placeholder)", icon="👥")
st.page_link("pages/4_🤝_Partners_View.py", label="Open Partners View (placeholder)", icon="🤝")
