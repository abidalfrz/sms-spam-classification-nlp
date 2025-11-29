import nltk
import pickle
import numpy as np
import emoji
import re
import ftfy
import unicodedata
import string

def demojize_emoji(text):
    emoji_pattern = re.compile(
        "["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE
       )
    
    return emoji_pattern.sub(lambda x: re.sub(r'[\W_]', " ",emoji.demojize(x.group())), text).strip()

def normalize_unicode(text):
    text = ftfy.fix_text(text)
    return "".join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def extract_domain_no_suffix(url):
    match = re.search(r'(?:https?:\/\/)?(?:www\.)?([^\.\/]+)', url)
    return match.group(1) if match else None

def cleaned_text(text):
    text = text.lower()
    text = demojize_emoji(text)
    text = normalize_unicode(text)

    slang_dict={
        "ga": "tidak",
        "gak": "tidak",
        "yg": "yang",
        "dgn": "dengan",
        "dg": "dengan",
        "klo": "kalau",
        "kalo": "kalau",
        "tdk": "tidak",
        "tlg": "tolong",
        "jgn": "jangan",
        "sdh": "sudah",
        "sbg": "sebagai",
        "bgt": "banget",
        "kmrn": "kemarin",
        "skrg":  "sekarang",
        "smoga": "semoga",
        "sy": "saya",
        "hrs": "harus",
        "dlm": "dalam",
        "ttp": "tetap",
        "krn": "karena",
        "dr": "dari",
        "pdhl": "padahal",
        "jd": "jadi",
        "aja": "saja",
        "ak": "saya",
        "gw": "saya",
        "lu": "kamu",
        "lo": "kamu",
        "gua": "saya",
        "dpt": "dapat",
        "bbrp": "beberapa",
        "gtw": "gatau",
        "gtau":"gatau",
        "gatau": "tidak tahu",
        "kpn": "kapan",
        "jt": "juta",
        "rb": "ribu",
        "hr": "hari",

    }

    for slang, formal in slang_dict.items():
        text = re.sub(rf'\b{slang}\b', formal, text)

    # take domain names 
    text = re.sub(r'http\S+|www\S+|https\S+', lambda x: extract_domain_no_suffix(x.group()), text, flags=re.MULTILINE)


    text = re.sub(rf'[{string.punctuation}]', ' ', text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # reduce repeated characters to two
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(f'[\n\r\t]', ' ', text)
    return text.strip()