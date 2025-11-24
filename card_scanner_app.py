"""
名片辨識系統 - Streamlit 網頁版 (最終修復版：延遲載入)
支援手機和電腦使用 + 圖片旋轉功能

執行方式:
python -m streamlit run card_scanner_app.py
"""

import streamlit as st
import easyocr
from PIL import Image
import cv2
import numpy as np
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import io


st.set_page_config(
    page_title="名片辨識系統",
    page_icon="📇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {padding: 1rem;}
    /* Streamlit 提示：'use_container_width' 將被移除，建議替換為 'width' */
    .stButton>button {width: 100%; height: 3rem; font-size: 1.2rem;}
    h1 {font-size: 1.8rem !important;}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# ⚠️ 這裡已移除 load_ocr_reader 函數和 @st.cache_resource
# 避免 App 啟動時記憶體崩潰
# =========================================================================


class BusinessCardScanner:
    def __init__(self, reader):
        self.reader = reader
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}',
            'website': r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?',
            'linkedin': r'linkedin\.com/in/[a-zA-Z0-9-]+',
        }
    
    def preprocess_image(self, image):
        img_array = np.array(image)
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # 使用 Denoising 和 CLAHE 進行圖像增強
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        return enhanced
    
    def extract_text(self, image):
        processed_img = self.preprocess_image(image)
        results = self.reader.readtext(processed_img)
        text_lines = [result[1] for result in results]
        text = '\n'.join(text_lines)
        return text, results
    
    def parse_info(self, text):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        info = {
            'name': '', 'title': '', 'company': '', 'email': '',
            'phone': '', 'mobile': '', 'website': '', 'address': '', 'linkedin': '', 'country': ''
        }
        
        # 1. 提取標準格式資訊
        email_match = re.search(self.patterns['email'], text, re.IGNORECASE)
        if email_match:
            info['email'] = email_match.group()
        
        phone_matches = re.findall(self.patterns['phone'], text)
        if phone_matches:
            valid_phones = [p for p in phone_matches if len(re.sub(r'[^\d]', '', p)) >= 8]
            if len(valid_phones) >= 1:
                info['phone'] = valid_phones[0]
            if len(valid_phones) >= 2:
                info['mobile'] = valid_phones[1]
        
        website_match = re.search(self.patterns['website'], text, re.IGNORECASE)
        if website_match:
            info['website'] = website_match.group()
        
        linkedin_match = re.search(self.patterns['linkedin'], text, re.IGNORECASE)
        if linkedin_match:
            info['linkedin'] = linkedin_match.group()
        
        # 2. 猜測非標準格式資訊 (姓名, 職稱, 公司)
        non_contact_lines = []
        for line in lines:
            if not any([
                re.search(self.patterns['email'], line, re.IGNORECASE),
                re.search(self.patterns['phone'], line),
                re.search(self.patterns['website'], line, re.IGNORECASE),
                'tel:' in line.lower(), 'fax:' in line.lower(),
            ]):
                non_contact_lines.append(line)
        
        if len(non_contact_lines) >= 1:
            info['name'] = non_contact_lines[0]
        
        if len(non_contact_lines) >= 2:
            potential_title = non_contact_lines[1]
            title_keywords = ['manager', 'director', 'ceo', 'cto', 'president', 
                            'executive', 'officer', 'head', 'lead', 'engineer',
                            'consultant', 'specialist', 'coordinator', 'supervisor']
            if any(keyword in potential_title.lower() for keyword in title_keywords):
                info['title'] = potential_title
            else:
                info['company'] = potential_title
        
        if len(non_contact_lines) >= 3 and not info['company
