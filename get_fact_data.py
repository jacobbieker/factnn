import os, sys
import requests
import yaml

with open("envs.yaml", "r") as credential_file:
    credentials = yaml.load(credential_file)

frontpage = requests.get(credentials['fact']['url'], auth=(credentials['fact']['username'], credentials['fact']['password']))
print(frontpage.text)



