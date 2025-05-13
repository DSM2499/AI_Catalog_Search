
import base64
import streamlit as st

def render_product_card(row):
    try:
        with open(row["image_path"], "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        
        img_html = f"""
        <div style='text-align: center; margin-bottom: 30px;'>
            <img src='data:image/jpeg;base64,{encoded}' 
                 style='width:100%; max-width:300px; border-radius: 10px;
                        transition: transform 0.3s ease-in-out;' 
                 onmouseover='this.style.transform="scale(1.05)"' 
                 onmouseout='this.style.transform="scale(1)"'/>
            <p style='margin-top: 10px;'><b>{row['title']}</b></p>
            <p style='font-size: 14px; color: #555;'>
                Category: {row['category'].capitalize()}<br>
                Color: {row['color'].capitalize()}<br>
                Gender: {row['gender'].capitalize()}<br>
                Usage: {row['usage'].capitalize()}
            </p>
        </div>
        """
        st.markdown(img_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load image for ID {row['id']}: {e}")
