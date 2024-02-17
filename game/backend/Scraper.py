import io
import requests
from pdfminer.high_level import extract_text

def download_pdf(url):
    """
    Download a PDF from a URL and return it as a bytes object.
    """
    response = requests.get(url)
    response.raise_for_status()  # This will ensure an HTTPError is raised for bad responses.
    return io.BytesIO(response.content)

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extract text from a PDF bytes object using pdfminer.six.
    """
    return extract_text(pdf_bytes)

# URL of the PDF you want to scrape
pdf_url = "https://web.stanford.edu/dept/cs_edu/resources/textbook/Reader-Beta-2012.pdf"

# Download the PDF
pdf_bytes = download_pdf(pdf_url)

# Extract text from the downloaded PDF
text = extract_text_from_pdf_bytes(pdf_bytes)

# Print the extracted text
print(text)
 
