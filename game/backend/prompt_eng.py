from bs4 import BeautifulSoup

import requests

page_to_scrape = requests.get("https://web.stanford.edu/class/cs106b/lectures/02-cpp/")