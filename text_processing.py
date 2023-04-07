import pandas as pd
import numpy as np
import re
# import os
# import argparse 
from datetime import date
import emoji

class text_processing:
    
    def __init__(self):
        pass

    def de_emojify(self, text):
        try:
            edited = emoji.demojize(text)
        except:
            print("emoji: ",text)
            edited = text
        return edited

    def remove_whitespace(self, text):
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

    def process_data(self, df, headers = ['title', 'subreddit', 'body']):
        df.dropna(inplace = True)
        for header in headers:
            df[header] = df[header].apply(lambda text: self.de_emojify(text))
            df[header] = df[header].apply(lambda text: self.remove_whitespace(text))
            df[header].replace('', np.nan, inplace=True)
        df.dropna(inplace = True)
        return df

    def save_output(self, df, append_name):
        today = date.today()
        str_date = today.strftime("%y%m%d")
        filename_body = str_date + "_" +append_name+".csv"
        df.to_csv(filename_body, index=False)
