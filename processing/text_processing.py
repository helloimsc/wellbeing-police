import pandas as pd
import numpy as np
import re
import os
import argparse 
from datetime import date

## Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="csv to parse")
args = parser.parse_args()
print("reading: " + args.filename)

filename = args.filename

def de_emojify(text):
    # print(text)
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\u200b"
                           "]+", flags = re.UNICODE)
    try:
        edited = regrex_pattern.sub(r' ',text)
    except:
        print("emoji: ",text)
        edited = text
    return edited

def remove_whitespace(text):
    try:
        edited = text.strip()
        edited = edited.lower()
        regex_badchars = re.compile(pattern=r"(&#x200B;)|\\|\*+|\.{2,}|…|\S*https?:\S*|\s*@\S{2,}", flags=re.IGNORECASE)
        regex_whitespace = re.compile(pattern=r" {2,}", flags=re.IGNORECASE)
        regex_newline = re.compile(pattern="\n|\t", flags=re.IGNORECASE)
        edited = regex_badchars.sub(r'', edited)
        edited = regex_newline.sub(r' ', edited)
        edited = regex_whitespace.sub(r' ', edited)
    except:
        print("whitespace: ",text)
        edited = text
    
    return edited

def process_data(df):
    df.dropna(inplace = True)
    df = df.map(lambda text: de_emojify(text))
    df = df.map(lambda text: remove_whitespace(text))
    df.replace('', np.nan, inplace=True)
    df_out = df.dropna()
    return df_out

def save_output(df, append_name):
    today = date.today()
    str_date = today.strftime("%y%m%d")
    filename_body = str_date + "_" +append_name+".csv"
    df.to_csv(filename_body, index=False)

if __name__ == "__main__":
    #get body 
    posts = pd.read_csv(filename, encoding="utf-8")
    body = posts['body']
    body_2 = process_data(body)
    print(body_2)
    save_output(body_2, "BipolarReddit_1000_body_noemoji")