import requests
import os
import time
import csv
import json

# Access your Pexels API Key directly
API_KEY = '################'  # Replace with your actual API key

# Define search queries for each theme
search_queries = {
    'urban_greenery': 'urban greenery',
    'natural_habitats': 'natural habitats',
    'eco_friendly_practices': 'eco-friendly practices'
}

# Number of images to download per theme
images_per_theme = 30  # Adjust as needed

def download_images(theme, query, count):
    headers = {
        'Authorization': API_KEY,
        'User-Agent': '###', #replace with your personal key 
    }
    per_page = 30  # Maximum per page is 80
    total_pages = (count // per_page) + 1

    os.makedirs(f'dataset/{theme}', exist_ok=True)
    image_count = 0

    # Prepare metadata file
    metadata_file = f'dataset/{theme}/metadata_pexels.csv'
    metadata_fields = [
        'image_id',
        'image_url',
        'photographer_name',
        'photographer_url',
        'source_platform',
        'theme'
    ]
    with open(metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metadata_fields)
        writer.writeheader()

    for page in range(1, total_pages + 1):
        params = {
            'query': query,
            'per_page': per_page,  # Must be between 1 and 80
            'page': page,
            'orientation': 'landscape',  # Optional parameter
        }
        response = requests.get(
            'https://api.pexels.com/v1/search',
            headers=headers,
            params=params
        )

        print(f"Request URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
            except json.decoder.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response content: {response.text}")
                return  # Exit the function or handle the error as needed
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.text}")
            return

        if 'photos' not in data:
            print(f"No 'photos' key in response data for theme '{theme}'. Response data: {data}")
            break

        for photo in data['photos']:
            image_id = photo['id']
            # Use the 'src' URL to download the image
            image_url = photo['src']['large']  # You can choose 'original', 'large', 'medium', etc.
            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                # Save image
                with open(f'dataset/{theme}/{image_id}_pexels.jpg', 'wb') as f:
                    f.write(image_response.content)
                image_count += 1
                print(f"Downloaded {image_count}/{count} images for {theme}")

                # Save metadata
                with open(metadata_file, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=metadata_fields)
                    writer.writerow({
                        'image_id': image_id,
                        'image_url': image_url,
                        'photographer_name': photo['photographer'],
                        'photographer_url': photo['photographer_url'],
                        'source_platform': 'Pexels',
                        'theme': theme
                    })
            else:
                print(f"Failed to download image {image_id}. Status code: {image_response.status_code}")

            if image_count >= count:
                break

        if image_count >= count:
            break

        # Respect API rate limits
        time.sleep(1)  # Adjust as needed

    print(f"Completed downloading images for {theme}")

# Main execution
if __name__ == "__main__":
    for theme, query in search_queries.items():
        download_images(theme, query, images_per_theme)
