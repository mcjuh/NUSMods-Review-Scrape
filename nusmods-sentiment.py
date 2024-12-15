from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium import webdriver
import re
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import requests
import csv

# Fetching module list
url = "https://api.nusmods.com/v2/2023-2024/moduleList.json" # Change the AY if necessary
response = requests.get(url)
response.raise_for_status() 
modules = response.json()
module_codes = [module["moduleCode"] for module in modules]

# Firefox Driver setup 
options = Options()
options.add_argument("-headless")
firefox_profile = FirefoxProfile()
firefox_profile.set_preference("javascript.enabled", True)
options.profile = firefox_profile
driver = webdriver.Firefox(options=options)

# Sentiment Analysis Setup - Change or add models to your preference
tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')
nlp = pipeline('sentiment-analysis', model = model, tokenizer = tokenizer)  

# Data collection preparation
data = {
    "Module Code": [],
    "Positive Comments": [],
    "Negative Comments": [],
    "Positive Sentiment Ratio": [],
    "Aggregated Sentiment Score": []
}

# Retrieving Disqus comments
for module_code in module_codes:
    try:
        driver.get("https://nusmods.com/courses/CM3141")
        WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//*[contains(@id,'dsq-app')]"))) # Important
        
        # Check if there are no posts
        st = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="no-posts"]'))).get_attribute('style')
        if "block" in st:
            data["Module Code"].append(module_code)
            data["Positive Comments"].append(None)
            data["Negative Comments"].append(None)
            data["Positive Sentiment Ratio"].append(None)
            data["Aggregated Sentiment Score"].append(None)
            print(f'No reviews for module: {module_code}')
            continue
        else:
            raw_reviews = [my_elem.text for my_elem in WebDriverWait(driver, 15)
                        .until(EC.visibility_of_all_elements_located((By.XPATH, "//ul[starts-with(@class, 'post-list')]")))]

            # Cleaning comments
            reviews = raw_reviews[0].split('Reply')[:-1]
            for i, r in enumerate(reviews):
                nr = r.strip()
                nr = re.sub(r'\s*see more\s*', '', nr)[2:-4]
                if not nr.startswith('NUSMods Mod'):
                    nr = nr[2:]
                reviews[i] = nr
            
            # Processing sentiment - Change or add parameters if needed
            results = nlp(reviews, truncation = True, max_length = 512) # Token limit is 512 for this model

            num_pos, num_neg, sum_score = 0, 0, 0
            for text, result in zip(reviews, results):
                if result['label'] == 'POSITIVE': 
                    num_pos += 1
                    sum_score += result['score']
                else:
                    num_neg += 1
                    sum_score -= result['score']

            pos_ratio = num_pos / (num_pos + num_neg)       
            agg_score = sum_score / (num_pos + num_neg)
            
            data["Module Code"].append(module_code)
            data["Positive Comments"].append(num_pos)
            data["Negative Comments"].append(num_neg)
            data["Positive Sentiment Ratio"].append(pos_ratio)
            data["Aggregated Sentiment Score"].append(agg_score)
            print(f'Finished processing: {module_code}')
    except Exception as e:
        print(f'Error occurred while processing module {module_code}: {e}')
driver.quit()

# Export data as csv
output_file = 'nus_module_sentiments.csv'
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(data.keys())
    writer.writerows(zip(*data.values()))

