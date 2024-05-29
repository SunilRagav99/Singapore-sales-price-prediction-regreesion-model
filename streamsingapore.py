import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import statistics
import pandas as pd
from streamlit_lottie import st_lottie
from PIL import Image
import json
import pickle
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderServiceError
from io import StringIO
import requests

data = pd.read_csv("mrt_lrt_data.csv")
mrt_location = pd.DataFrame(data)


drive_link = r"https://drive.google.com/uc?id=1xLiNiDdNc1NJ1TctNqYusgb1nsrj8bG6&export=download"
response = requests.get(drive_link)
csv_data = StringIO(response.text)
datacombine = pd.read_csv(csv_data)

# datacombine = pd.read_csv("combined.csv")



# datacombine = pd.read_csv(r"C:\Users\sunil\OneDrive\Desktop\python guvi\singapore prediction\combined.csv")
ad = datacombine["address"].unique()
street = datacombine["street_name"].unique()

icon = Image.open("8345929.png")
with open("Animation - 1716999046955.json", 'r') as f:
    data = json.load(f)

with open("Animation - 1717014736285.json", 'r') as ff:
    data1 = json.load(ff)

st.set_page_config(page_title="Singapore Resale Flat Price Prediction | By SUNIL RAGAV",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center; color:#407ECD;'>Singapore Resales Flat Price Prediction</h1>", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "blue"},
                                   "icon": {"font-size": "10px"}
                                   })

if selected == "About Project":
    col1, col2, col3 = st.columns([3, 1, 2])
    with col1:
        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
        st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                    "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                    "Model Deployment")
        st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                    "it as a user-friendly online application in order to provide accurate predictions about the "
                    "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                    "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                    "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                    "of criteria, including location, the kind of apartment, the total square footage, and the length "
                    "of the lease. The provision of customers with an expected resale price based on these criteria is "
                    "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
        st.markdown("### :blue[Domain :] Real Estate")
    with col3:
        st_lottie(data, reverse=True, height=600, width=400, speed=1, loop=True, quality='high', key='spinner1')

if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task,Accuracy:82%)]")

    with st.form("form1"):
        col1,col2,col3=st.columns([3,1,2])
        with col1:
       
            street_name = st.selectbox("Street Name", street)
            block = st.text_input("Block Number")
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence year')
            storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")
        with col3:
            st_lottie(data1, reverse=True, height=400, width=400, speed=1, loop=True, quality='high', key='spinner2')

if submit_button:
            with open("singamodel.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open("singascaler.pkl", 'rb') as f:
                scaler_loaded = pickle.load(f)

            
            lease_remain_years = 99 - (2024 - lease_commence_date)

           
            split_list = storey_range.split(' TO ')
            float_list = [float(i) for i in split_list]
            storey_median = statistics.median(float_list)

           
            address = block + " " + street_name + " " + "Singapore"
            st.write("Address:",address)

            geolocator = Nominatim(user_agent="geo")

            try:
                location = geolocator.geocode(address)
                if location:
                    resp = (location.latitude, location.longitude)
                    st.write("coordinates:",resp)

                    
                    mrt_lat = mrt_location['lat']
                    mrt_long = mrt_location['lng']
                    list_of_mrt_coordinates = list(zip(mrt_lat, mrt_long))

                    # st.write(list_of_mrt_coordinates)

                    
                    list_of_dist_mrt = []

                    for mrt_coord in list_of_mrt_coordinates:
                        try:
                            distance = geodesic(resp, mrt_coord).meters
                            list_of_dist_mrt.append(distance)
                        except ValueError as e:
                            st.error(f"Error calculating distance: {e}")
                            continue

                    if list_of_dist_mrt:
                        min_dist_mrt = min(list_of_dist_mrt)
                    else:
                        st.error("No valid MRT coordinates provided")
                        min_dist_mrt = None

                    # st.write("Minimum distance to an MRT station:", min_dist_mrt)

                    
                    cbd_dist = geodesic(resp, (1.2830, 103.8513)).meters  # CBD coordinates

                    
                    if min_dist_mrt is not None:
                        new_sample = np.array(
                            [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                        # st.write(cbd_dist)
                        # st.write(min_dist_mrt)
                        # st.write(floor_area_sqm)
                        # st.write(lease_remain_years)
                        # st.write(storey_median)
                        new_sample = scaler_loaded.transform(new_sample)
                        new_pred = loaded_model.predict(new_sample)
                        st.write('## :green[Predicted resale price:] ',new_pred[0])
                else:
                    st.error("Address could not be geocoded. Please check the address entered.")
            except GeocoderServiceError as e:
                st.error(f"Geocoding service error: {e}")

