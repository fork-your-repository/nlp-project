

import requests
import json
from typing import List, Dict, Union, cast

class GitHubAPIError(Exception):
    def __init__(self, status_code, response_data):
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(f"Error response from GitHub API! Status code: {status_code}, Response: {response_data}")

def github_api_request(url: str, headers: dict) -> Union[List, Dict]:
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError if the response status is not 2xx
        response_data = response.json()  # Parse the JSON response
        return response_data
    except requests.exceptions.RequestException as e:
        raise e  # Re-raise network-related errors for upper-level handling
    except requests.exceptions.HTTPError as e:
        raise GitHubAPIError(response.status_code, response.json())  # Raise custom error for non-2xx status codes

def get_repo_language(repo_info: Dict) -> str:
    # Retrieve the "language" key from the repository info, or return an empty string if it doesn't exist
    return repo_info.get("language", "")

def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)  # Get repository contents using the GitHub API
    if isinstance(contents, list):
        return contents  # Return the contents if they are a list
    raise GitHubAPIError(-1, contents)  # Raise custom error for non-list responses

def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    for file in files:
        if file["name"].lower().startswith("readme"):  # Look for a file whose name starts with "readme"
            return file["download_url"]  # Return the download URL of the readme file
    return ""  # Return an empty string if no matching readme file is found

def process_repo(repo: str) -> Dict[str, str]:
    contents = get_repo_contents(repo)  # Get the contents of the repository
    readme_contents = requests.get(get_readme_download_url(contents)).text  # Get and read the readme contents
    repo_info = github_api_request(f"https://api.github.com/repos/{repo}", {})  # Get repository info
    return {
        "repo": repo,
        "language": get_repo_language(repo_info),
        "readme_contents": readme_contents,
    }

REPOS = [...]  # List of repository names

if __name__ == "__main__":
    # Process each repository and collect the data
    data = [process_repo(repo) for repo in REPOS]
    
    # Write the collected data to a JSON file with proper formatting
    with open("data2.json", "w") as json_file:
        json.dump(data, json_file, indent=1)


# # Example usage
# github_url = "https://api.github.com/repos/username/repo"
# github_headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}

# try:
#     response_data = github_api_request(github_url, github_headers)
#     print("API response:", response_data)  # Print the API response
# except GitHubAPIError as e:
#     print("API error:", e)  # Print the custom error message for GitHub API errors
# except requests.exceptions.RequestException as e:
#     print("Network error:", e)  # Print the error message for network-related errors