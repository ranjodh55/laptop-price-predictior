import numpy as np
import streamlit as st
import pickle

# load the models
df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

# title of the page
st.title('Laptop Price Predictor')

# set different dropdowns
company = st.selectbox('Brand', np.sort(df['Company'].unique()))
lp_type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', np.sort(df['Ram'].unique()))
weight = st.number_input('Weight of the laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                           '2560x1440', '2304x1440'])
screen_size = st.selectbox('Screen Size',
                      [11.6, 12, 13.3, 14, 15.6, 17])
cpu = st.selectbox('CPU', np.sort(df['cpu'].unique()))
hdd = st.selectbox('HDD (in GB)', np.sort(df['hdd'].unique()))
ssd = st.selectbox('SSD (in GB)', np.sort(df['ssd'].unique()))
gpu = st.selectbox('GPU', np.sort(df['gpu'].unique()))
os = st.selectbox('Operating System', np.sort(df['os'].unique()))

if st.button('Predict Price'):

    res = resolution.split('x')
    width = int(res[0])
    height = int(res[1])

    if weight == 0.00 or weight < 1.00:
        st.warning('Weight cannot be 0 or less than 1Kg', icon="⚠️")
    else:
        ppi = np.sqrt((width ** 2) + (height ** 2)) / weight
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0
        query = np.array([company, lp_type, ram, weight, ips, touchscreen, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
        query = query.reshape(1, 12)
        pred = np.exp(pipe.predict(query))
        price = round(pred[0], 2)
        st.header(f'The price of the laptop should be around Rs.{price}')