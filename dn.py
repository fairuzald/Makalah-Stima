from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_reviews(url):
    # Setup Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Menjalankan browser dalam mode headless
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Mengunjungi halaman produk
    driver.get(url)
    
    # Tunggu beberapa detik untuk memastikan semua konten dimuat
    driver.implicitly_wait(10)
    
    # Memuat konten halaman
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Mencari semua elemen div dengan kelas 'shopee-product-rating' atau data-testid yang sesuai
    reviews = soup.find_all('span', {'data-testid': 'lbmItemUlasan'})
    
    # Mengambil teks dari setiap elemen ulasan
    review_texts = [review.text.strip() for review in reviews]
    
    # Menutup browser
    driver.quit()
    
    return review_texts

# URL halaman produk Shopee
url = 'https://www.tokopedia.com/liger-official/liger-handsfree-headset-earphone-l-10-metal-stereo-bass-putih?source=homepage.top_carousel.0.39123'  # Ganti dengan URL yang sesuai

# Memanggil fungsi untuk mengambil ulasan
reviews = scrape_reviews(url)
print(reviews)
# Mencetak hasil ulasan
for i, review in enumerate(reviews, 1):
    print(f"Review {i}: {review}")


# print(f"Scraped {len(reviews)} reviews.")
# from transformers import pipeline

# # Menginisialisasi model sentiment analysis
# sentiment_pipeline = pipeline('sentiment-analysis')

# # Analisis sentimen langsung
# def analyze_sentiment_no_dc(reviews):
#     sentiments = sentiment_pipeline(reviews)
#     return sentiments

# no_dc_sentiments = analyze_sentiment_no_dc(reviews)
# print(no_dc_sentiments)
# import concurrent.futures

# # Fungsi untuk analisis sentimen per subset
# def analyze_sentiment_subset(subset_reviews):
#     return sentiment_pipeline(subset_reviews)

# # Pembagian data ulasan menjadi subset
# def divide_and_conquer_analysis(reviews, num_subsets=4):
#     subset_size = len(reviews) // num_subsets
#     subsets = [reviews[i * subset_size:(i + 1) * subset_size] for i in range(num_subsets)]
    
#     # Analisis sentimen secara paralel
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_subset = {executor.submit(analyze_sentiment_subset, subset): subset for subset in subsets}
#         results = []
#         for future in concurrent.futures.as_completed(future_to_subset):
#             results.extend(future.result())
#     return results

# dc_sentiments = divide_and_conquer_analysis(reviews)
# print(dc_sentiments)
# # Fungsi untuk menghitung sentimen keseluruhan
# def calculate_overall_sentiment(sentiments):
#     positive = sum(1 for sentiment in sentiments if sentiment['label'] == 'POSITIVE')
#     negative = sum(1 for sentiment in sentiments if sentiment['label'] == 'NEGATIVE')
#     return {'positive': positive, 'negative': negative}

# no_dc_overall_sentiment = calculate_overall_sentiment(no_dc_sentiments)
# dc_overall_sentiment = calculate_overall_sentiment(dc_sentiments)

# print("Overall Sentiment without Divide and Conquer:", no_dc_overall_sentiment)
# print("Overall Sentiment with Divide and Conquer:", dc_overall_sentiment)
