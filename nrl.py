from bs4 import BeautifulSoup
import requests
import lxml
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import streamlit as st
import numpy as np
from requests_html import HTMLSession




# https://stackoverflow.com/questions/55351871/beautifulsoup-attributeerror-nonetype-object-has-no-attribute-text
# https://github.com/njpowers7915/NRL-match-data/blob/master/scraping/2_stat_scraper_2018.ipynb

# st.write( [' + str(1') +'] )

# session = HTMLSession()
# response = session.get('https://www.nrl.com/draw/nrl-premiership/2021/finals-week-3/rabbitohs-v-sea-eagles/')
# soup = BeautifulSoup(response.content, 'html.parser')
# title =  response.html.xpath(('//*[@id="player-stats"]/div[2]/div/div[3]/div/table/tbody/tr[2]/td[2]/a'))


# driver = webdriver.Chrome(ChromeDriverManager().install())
# name_field = driver.find_element_by_xpath(('//*[@id="player-stats"]/div[2]/div/div[3]/div/table/tbody/tr[1]/td[2]/a')).get_attribute('innerText').strip()
# st.write(name_field)
# st.write(title)

# tomorrow_weather = soup.find(id="tabs-match-centre-")
# st.write(tomorrow_weather)



# URL = "https://www.nrl.com/draw/nrl-premiership/2021/finals-week-3/rabbitohs-v-sea-eagles/"
# # URL = "https://realpython.github.io/fake-jobs/"
# page = requests.get(URL)
# soup = BeautifulSoup(page.content, "html.parser")
# # results = soup.find(id="ResultsContainer")
# results = soup.find(id="tabs-match-centre-")

# # st.write(page.text)
# st.write(results.prettify())