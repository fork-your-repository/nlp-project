# Import Libraries
import pandas as pd
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests 
from bs4 import BeautifulSoup
from env_miatta import github_token, github_username
import time

# ACQUIRE

REPOS = ['TheAlgorithms/Python',
 'apache/flink',
 'forezp/SpringCloudLearning',
 'learn-co-students/python-dictionaries-readme-data-science-intro-000',
 'angular/angular-phonecat',
 'bloominstituteoftechnology/github-usercard',
 'learn-co-students/javascript-arrays-lab-bootcamp-prep-000',
 'tastejs/todomvc',
 'learn-co-students/jupyter-notebook-introduction-data-science-intro-000',
 'hasura-imad/imad-2016-app',
 'josephmisiti/awesome-machine-learning',
 'RasaHQ/rasa',
 'learn-co-students/javascript-intro-to-functions-lab-bootcamp-prep-000',
 'CorentinJ/Real-Time-Voice-Cloning',
 'vivienzou1/DL-Notes-for-interview',
 'MakeSchool/FlappyBirdTemplate-Spritebuilder',
 'jtleek/datasharing',
 'jquery/jquery',
 'freefq/free',
 'moby/moby',
 'learn-co-students/js-deli-counter-bootcamp-prep-000',
 'bloominstituteoftechnology/React-Todo',
 '996icu/996.ICU',
 'modood/Administrative-divisions-of-China',
 'codefresh-contrib/gitops-certification-examples',
 'TheOdinProject/javascript-exercises',
 'ColorlibHQ/gentelella',
 'learn-co-students/python-variables-lab-data-science-intro-000',
 'vaxilu/x-ui',
 'linuxacademy/cicd-pipeline-train-schedule-dockerdeploy',
 'RedHatTRaining/DO288-apps',
 'luchihoratiu/debug-via-ssh',
 'deadlyvipers/dojo_rules',
 'jenkinsci/jenkins',
 'mqyqingfeng/Blog',
 'spring-projects/spring-framework',
 'apache/kafka',
 'learn-co-curriculum/react-hooks-lists-and-keys-lab',
 '233boy/v2ray',
 'typicode/json-server',
 'learn-co-students/js-beatles-loops-lab-bootcamp-prep-000',
 'Azure/azure-quickstart-templates',
 'learn-co-students/js-from-dom-to-node-bootcamp-prep-000',
 'trekhleb/homemade-machine-learning',
 'AtsushiSakai/PythonRobotics',
 'xitu/gold-miner',
 'xingshaocheng/architect-awesome',
 'celery/celery',
 'ibm-developer-skills-network/xzceb-flask_eng_fr',
 'lazyprogrammer/machine_learning_examples',
 'ripienaar/free-for-dev',
 'jeecgboot/jeecg-boot',
 'bloominstituteoftechnology/Preprocessing-II',
 'Turonk/infra_actions',
 'reduxjs/redux',
 'rapid7/metasploit-framework',
 'bloominstituteoftechnology/node-db1-project',
 'nightscout/cgm-remote-monitor',
 'alx-tools/your_first_code',
 'dcxy/learngit',
 'brentley/ecsdemo-nodejs',
 'yankouskia/additional_5',
 'altercation/solarized',
 'supabase/supabase',
 'Ebazhanov/linkedin-skill-assessments-quizzes',
 'travis-ci/docs-travis-ci-com',
 'learn-co-students/python-lists-lab-data-science-intro-000',
 'stacksimplify/azure-aks-kubernetes-masterclass',
 'mitmproxy/mitmproxy',
 'jumpserver/jumpserver',
 'scutan90/DeepLearning-500-questions',
 'ultralytics/yolov5',
 'forem/forem',
 'bloominstituteoftechnology/module-challenge-intro-to-git',
 'jackfrued/Python-100-Days',
 'docsifyjs/docsify',
 'heartcombo/devise',
 'linuxacademy/cicd-pipeline-train-schedule-gradle',
 'othneildrew/Best-README-Template',
 'developerforce/intro-to-heroku',
 'kodekloudhub/certified-kubernetes-administrator-course',
 'ceph/ceph',
 'bilibili/ijkplayer',
 'taizilongxu/interview_python',
 'ant-design/ant-design-pro',
 'progedu/adding-up',
 'micropython/micropython',
 'XX-net/XX-Net',
 'cyclic-software/starter-express-api',
 'bloominstituteoftechnology/team-builder',
 'bloominstituteoftechnology/User-Interface',
 'JetBrains/kotlin',
 'learn-co-students/javascript-logging-lab-bootcamp-prep-000',
 'barryclark/jekyll-now',
 'raulmur/ORB_SLAM2',
 'amjuarez/bytecoin',
 'trustwallet/assets',
 'Binaryify/NeteaseCloudMusicApi',
 'vuejs/v2.vuejs.org',
 'keycloak/keycloak',
 'thingsboard/thingsboard',
 'learn-co-curriculum/phase-0-html-issue-bot-9000-lab',
 'qmk/qmk_firmware',
 'learn-co-curriculum/phase-0-the-dom-editing-lab',
 'PowerShellMafia/PowerSploit',
 'devopshydclub/vprofile-project',
 'bloominstituteoftechnology/DOM-II',
 'zhisheng17/flink-learning',
 'protocolbuffers/protobuf',
 'GitbookIo/gitbook',
 'selectize/selectize.js',
 'kubernetes/kubernetes',
 'facebookresearch/fastText',
 'espressif/esp-idf',
 'lin-xin/vue-manage-system',
 'Significant-Gravitas/Auto-GPT',
 'torvalds/linux',
 'namndwebdev/tang-crush'] 
         
# The REPOS list was made into 23 json files in batches of 5-6 (due to 404 error issues) and combined into 1 json named data2.json.
         

# Function to make a GitHub API request
def github_api_request(url: str) -> Union[List, Dict]:
    """
    Makes a request to the GitHub API and returns the JSON response.
    """
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from GitHub API! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data

# Function to get the programming language of a repository
def get_repo_language(repo: str) -> str:
    """
    Retrieves the programming language of a GitHub repository.
    """
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )

# Function to get the contents of a repository
def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    """
    Retrieves the contents of a GitHub repository.
    """
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )

# Function to get the download URL for a repository's README file
def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the GitHub API that lists the files in a repo
    and returns the URL that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""

# Function to process repository information
def process_repo(repo: str) -> Dict[str, str]:
    """
    Processes a repository, retrieving its language and README contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(
        f"https://github.com/{repo}/blob/master/README.md"
    ).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }

# Function to scrape GitHub data for multiple repositories
def scrape_github_data() -> List[Dict[str, str]]:
    """
    Scrapes GitHub data for multiple repositories, processing each one.
    """
    processed_data = []
    for repo in REPOS:
        processed_repo = process_repo(repo)
        processed_data.append(processed_repo)
        time.sleep(60)  # Add a 60-second delay between requests
    return processed_data

if __name__ == "__main__":
    # Scrape GitHub data and save to a JSON file
    data = scrape_github_data()
    json.dump(data, open("data24.json", "w"), indent=1)


# List of JSON file names
json_file_names = [
    'data1.json',
    'data3.json',
    'data4.json',
    'data5.json',
    'data6.json',
    'data7.json',
    'data8.json',
    'data9.json',
    'data10.json',
    'data11.json',
    'data12.json',
    'data13.json',
    'data14.json',
    'data15.json',
    'data16.json',
    'data17.json',
    'data18.json',
    'data19.json',
    'data20.json',
    'data21.json',
    'data22.json',
    'data23.json',
    'data24.json'
]

# Combine JSON data from multiple files into a single list
combined_data = []

for file_name in json_file_names:
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
        combined_data.extend(data)

# Write the combined data to a new JSON file named "data2.json"
with open("data2.json", "w") as combined_json_file:
    json.dump(combined_data, combined_json_file, indent=1)
