import numpy as np
import pandas as pd
from faker import Faker
import random
import babel
from babel.dates import format_date
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging

fake = Faker()
Faker.seed(12345)
random.seed(12345)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']


LOCALES = ['en_US']

def generate_date_data():
    try:
        logging.info('generate_date_data')
        dt = fake.date_object()
        human_readable_dt = None
        machine_readable_dt = None
        try:
            human_readable_dt = format_date(dt, random.choice(FORMATS), "en_US")
            human_readable_dt = human_readable_dt.replace(",", "")
            machine_readable_dt = dt.isoformat()
        except AttributeError as e:
            return None, None, None
        return human_readable_dt, machine_readable_dt, dt
    except Exception as e:
        raise CustomException(e,sys)



def load_date_dataset(num_examples=100):    
    try:
        logging.info('load_date_dataset')
        dataset = []
        for row in range(num_examples):
            h_dt, m_dt, dt = generate_date_data()        
            dataset.append([h_dt, m_dt])    
        return np.array(dataset)
    except Exception as e:
        raise CustomException(e,sys)



dataset = load_date_dataset(12000)

df_dates = pd.DataFrame({"h_dt": dataset[:, 0], "m_dt": dataset[:, 1]})
df_dates.head(30)

# Save "h_dt" to CSV without column names
df_dates[['h_dt']].to_csv('artifacts/source.txt', sep='\t', index=False, header=False)

# Save "m_dt" to CSV without column names
df_dates[['m_dt']].to_csv('artifacts/translation.txt', sep='\t', index=False, header=False)
