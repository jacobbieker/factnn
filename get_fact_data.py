import os, sys
import requests
import yaml
from bs4 import BeautifulSoup

with open("envs.yaml", "r") as credential_file:
    credentials = yaml.load(credential_file)

frontpage = requests.get(credentials['fact']['url'], auth=(credentials['fact']['username'], credentials['fact']['password']))
print(frontpage)
frontpage_web = BeautifulSoup(frontpage.text, "lxml")
link_text = []
for link in frontpage_web.find_all('a', href=True):
    if link.get_text(strip=True):
        link_text.append(link['href'])

print(link_text)

observations_page = requests.get(credentials['fact']['baseurl']+link_text[0], auth=(credentials['fact']['username'], credentials['fact']['password']))
print(observations_page.content)
observations_web = BeautifulSoup(observations_page.text, "lxml")
obs_link_text = []
for link in observations_web.find_all('a', href=True):
    if link.get_text(strip=True):
        obs_link_text.append(link['href'])
print(obs_link_text)
simulation_page = requests.get(credentials['fact']['baseurl']+link_text[2], auth=(credentials['fact']['username'], credentials['fact']['password']))
simulation_web = BeautifulSoup(simulation_page.text, "lxml")
print(simulation_page.content)