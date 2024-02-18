from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Setup Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# URL of the page to scrape
url = "https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=d11e04a0-7981-4202-9fcc-b0de00034016&start=14.292547993878093"

# Use Selenium to get the page
driver.get(url)

# Wait for the dynamic content to load
time.sleep(1000)  # Adjust this sleep time according to your network speed and page response time

# Now that the page is loaded, get the page source
html = driver.page_source

# Parse the page source with BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Find elements containing the transcript text
# Adjust the selector according to the actual structure of the Panopto page
transcript_elements = soup.find_all('span',class_='event-text')  # This selector might need to be updated

# Extract and print the transcript text
print(transcript_elements)
for element in transcript_elements:
    print(element.get_text())

# Don't forget to close the browser
driver.quit()