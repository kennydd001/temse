import requests

def google_search(query, api_key, cse_id, num_results=10, country='BE'):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': cse_id,
        'key': api_key,
        'num': num_results,
        'cr': f'country{country}'  # Restricting search to Belgium
    }
    response = requests.get(search_url, params=params)
    return response.json()

def print_google_search_results():
    YOUR_GOOGLE_API_KEY = "AIzaSyB86-Q_E0xKMdmfEw5NeduMv49YX5VYLGk"
    YOUR_GOOGLE_CSE_ID = "81df4b402524848ea"
    query = "temse aldi openingsuren"

    results = google_search(query, YOUR_GOOGLE_API_KEY, YOUR_GOOGLE_CSE_ID)
    for i, item in enumerate(results.get("items", [])):
        print(f"Result {i+1}:")
        print(f"Title: {item['title']}")
        print(f"Snippet: {item['snippet']}")
        print(f"URL: {item['link']}\n")

# Execute the function to print results
print_google_search_results()

# Note: As this code involves network access and API requests, it won't run in this environment.
# You'll need to run it in your local environment with internet access.
