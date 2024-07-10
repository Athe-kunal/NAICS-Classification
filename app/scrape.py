from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import json 

def process_naics_text(txt,sup:bool=False):
    split_text = txt.split(" ",1)
    code = split_text[0].strip()
    industry_name, desc = split_text[1].split(":",1)
    desc = desc.strip()
    if sup:
        industry_name = industry_name[:-1]
    return {"code":code,"industry_name":industry_name,"industry_description":desc,"comparable":sup}


def get_naics_parents_child(driver:webdriver.Chrome):
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    naics_list = soup.find_all(class_="naics_list")

    nl_text_list = []
    for nl in naics_list:
        if nl.find("sup"):
            nl_text = process_naics_text(nl.text,True)
        else:
            nl_text = process_naics_text(nl.text,False)
        nl_text_list.append(nl_text)
    

    naics_dict = {}
    parent_naics = nl_text_list[0]

    naics_dict['parent_code'] = parent_naics['code']
    naics_dict['parent_industry_name'] = parent_naics['industry_name']
    naics_dict['parent_industry_description'] = parent_naics['industry_description']

    naics_dict['child_naics_dict'] = []

    curr_loop_industry = ""
    for child_naics in nl_text_list[1:]:
        curr_child_industry = child_naics['industry_name']
        # Append the industry codes if it's already there
        if curr_child_industry == curr_loop_industry:
            naics_dict['child_naics_dict'][-1]['child_code'].append(child_naics['code'])
            naics_dict['child_naics_dict'][-1]['child_description'] += " " + child_naics['industry_description']
        else:
            naics_dict['child_naics_dict'].append({
                "child_code": [child_naics['code']],
                "child_industry_name": curr_child_industry,
                "child_description": child_naics['industry_description']
            })
        curr_loop_industry = curr_child_industry
    return naics_dict

def save_naics(all_naics:list[dict]):
    json_object = json.dumps(all_naics, indent=4)
 
    # Writing to sample.json
    with open("app/NAICS.json", "w") as outfile:
        outfile.write(json_object)
    print("NAICS data saved successfully.")

def scrape_main():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("https://www.census.gov/naics/?58967?yearbck=2022")
    driver.maximize_window()

    # Click on 2022 section
    element = driver.find_element(By.ID, 'naics-2022-toggle-btn')
    element.click()
    time.sleep(1)
    industry_elements = driver.find_elements(By.XPATH, "//*[contains(@class, 'naics-sector-search-btn') and contains(@class, 'btn-link')]")

    all_naics = []
    for ie_range in range(len(industry_elements)):
        ie = driver.find_elements(By.XPATH, "//*[contains(@class, 'naics-sector-search-btn') and contains(@class, 'btn-link')]")[ie_range]
        ie.click()
        time.sleep(1)
        curr_naics_dict = get_naics_parents_child(driver)
        all_naics.append(curr_naics_dict)
        driver.execute_script("window.history.go(-1)")
        time.sleep(1)
    
    save_naics(all_naics)
    driver.quit()
