import random
from faker import Faker
import uuid
import string
import re
# Array nama menu makanan dan minuman
menu_makanan = [
    "Nasi Goreng", "Ayam Goreng", "Mie Goreng", "Sate Ayam", "Bakso", "Gado-Gado", "Nasi Kuning",
    "Rendang", "Soto Ayam", "Pecel Lele", "Martabak", "Nasi Uduk", "Sop Buntut", "Capcay", "Ikan Bakar",
    "Nasi Padang", "Mie Ayam", "Nasi Campur", "Nasi Tumpeng", "Rawon", "Bubur Ayam", "Sop Iga",
    "Sambal Goreng", "Ketoprak", "Lontong Sayur", "Tahu Goreng", "Tempe Goreng", "Kare Ayam",
    "Bebek Goreng", "Kwetiau Goreng", "Nasi Gudeg", "Iga Bakar", "Pisang Goreng", "Kroket",
    "Kangkung Belacan", "Ayam Bakar", "Kupat Tahu", "Mie Rebus", "Sambal Matah",
    "Laksa", "Ayam Rica-Rica", "Nasi Liwet", "Nasi Timbel", "Sop Kambing", "Pindang", "Ayam Betutu",
    "Bebek Betutu", "Sate Padang", "Nasi Goreng Jawa", "Nasi Goreng Seafood", "Coto Makassar",
    "Bubur Manado", "Rujak Cingur", "Sate Madura", "Sate Lilit", "Nasi Bakar"
]

menu_minuman = [
    "Es Teh", "Es Jeruk", "Jus Alpukat", "Es Cincau", "Jus Mangga", "Es Campur", "Es Doger", "Jus Apel",
    "Jus Melon", "Jus Nanas", "Es Kopyor", "Es Buah", "Es Kelapa Muda", "Jus Markisa", "Es Timun Suri",
    "Kopi Tubruk", "Kopi Susu", "Kopi Hitam", "Teh Tarik", "Teh Hangat", "Teh Poci", "Teh Manis", "Teh Botol",
    "Bandrek", "Susu Jahe", "Susu Kurma", "Soda Gembira", "Cendol", "Wedang Jahe", "Wedang Ronde",
    "Jus Wortel", "Es Krim", "Air Kelapa", "Sirup Marjan", "Sari Asem", "Bajigur", "Wedang Uwuh", "Jahe Wangi",
    "Teh Talua", "Susu Jahe Merah", "Wedang Secang", "Teh Susu", "Es Cincau Hitam", "Sari Kurma", "Jus Pepaya",
    "Kolang-Kaling", "Es Oyen", "Susu Kacang Hijau", "Es Cendol", "Air Zam-Zam", "Susu Kurma Muda",
    "Jamu Beras Kencur"
]

# Deskripsi untuk masing-masing menu makanan
deskripsi_makanan = {
    "Nasi Goreng": "Nasi yang digoreng dengan bumbu rempah dan disajikan dengan telur, daging ayam, dan sayuran.",
    "Ayam Goreng": "Potongan ayam yang digoreng garing dengan bumbu khas, disajikan dengan nasi dan sayuran.",
    "Mie Goreng": "Mie yang digoreng dengan bumbu rempah, dicampur dengan telur, sayuran, dan daging/ayam/udang.",
    "Sate Ayam": "Potongan daging ayam yang dipanggang dengan bumbu kacang, disajikan dengan lontong dan irisan bawang.",
    "Bakso": "Bola daging yang direbus dengan kuah kaldu sapi dan disajikan dengan mie, bihun, atau nasi.",
    "Gado-Gado": "Sayuran segar seperti kacang panjang, tauge, bayam, dan ketimun dengan saus kacang khas Indonesia.",
    "Nasi Kuning": "Nasi yang dimasak dengan kunyit, disajikan dengan telur, ayam goreng, tempe, sambal, dan acar.",
    "Rendang": "Daging sapi yang dimasak dalam santan dengan rempah-rempah khas Minang.",
    "Soto Ayam": "Sup ayam dengan kuah bening, disajikan dengan nasi, irisan telur, ketupat, dan bawang goreng.",
    "Pecel Lele": "Lele goreng yang disajikan dengan sambal pecel, lalapan, dan nasi.",
    "Martabak": "Adonan terigu yang digoreng dengan isian telur, daging/ayam, bawang, dan bumbu lainnya.",
    "Nasi Uduk": "Nasi yang dimasak dengan santan dan rempah, disajikan dengan lauk-pauk seperti ayam, tempe, dan sambal.",
    "Sop Buntut": "Sup buntut sapi dengan kuah bening yang kaya rempah, disajikan dengan nasi.",
    "Capcay": "Sayuran dan daging/ayam/udang yang diwok dengan bumbu-bumbu dan disajikan dengan nasi.",
    "Ikan Bakar": "Ikan yang dibakar dengan bumbu rempah, disajikan dengan sambal dan nasi.",
    "Nasi Padang": "Nasi dengan berbagai lauk khas Padang seperti rendang, gulai ayam, sambal, dan sayuran.",
    "Mie Ayam": "Mie dengan potongan ayam dan irisan pangsit, disajikan dengan kuah ayam dan bawang goreng.",
    "Nasi Campur": "Nasi dengan berbagai lauk seperti ayam, telur, tempe, tahu, dan sambal.",
    "Nasi Tumpeng": "Nasi berbentuk kerucut dengan lauk ayam, telur, tahu, tempe, sambal, dan kerupuk.",
    "Rawon": "Sup daging sapi dengan kuah hitam khas Jawa Timur, disajikan dengan nasi dan irisan telur.",
    "Bubur Ayam": "Bubur nasi dengan potongan ayam, bawang goreng, daun bawang, dan kerupuk.",
    "Sop Iga": "Sup iga sapi dengan kuah bening, disajikan dengan nasi dan sayuran.",
    "Sambal Goreng": "Sayuran seperti kacang panjang dan tahu yang digoreng dengan sambal, disajikan dengan nasi.",
    "Ketoprak": "Mie rebus dengan tauge, ketupat, telur, tahu, dan bumbu kacang, disajikan dengan sambal.",
    "Lontong Sayur": "Lontong dengan kuah sayur bening, disajikan dengan telur, tahu, tempe, dan sambal.",
    "Tahu Goreng": "Potongan tahu yang digoreng garing, disajikan dengan sambal dan kecap.",
    "Tempe Goreng": "Potongan tempe yang digoreng garing, disajikan dengan sambal dan kecap.",
    "Kare Ayam": "Potongan ayam yang dimasak dengan kuah kare yang kaya rempah, disajikan dengan nasi.",
    "Bebek Goreng": "Potongan bebek yang digoreng garing dengan bumbu khas, disajikan dengan nasi dan sayuran.",
    "Kwetiau Goreng": "Kwetiau yang digoreng dengan bumbu dan berbagai bahan seperti ayam, udang, sayuran, dan telur.",
    "Nasi Gudeg": "Nasi dengan lauk gudeg (nangka muda santan) dan sambal krecek.",
    "Iga Bakar": "Iga sapi yang dibakar dengan bumbu khas, disajikan dengan nasi dan sambal.",
    "Pisang Goreng": "Potongan pisang yang digoreng garing, bisa disajikan dengan keju atau sambal.",
    "Kroket": "Makanan yang terbuat dari kentang yang dihaluskan dan digoreng dengan berbagai isian.",
    "Kangkung Belacan": "Sayuran kangkung yang disajikan dengan sambal belacan.",
    "Ayam Bakar": "Potongan ayam yang dibakar dengan bumbu khas, disajikan dengan nasi dan sambal.",
    "Es Teler": "Minuman es campur dengan potongan buah, es serut, dan santan.",
    "Kupat Tahu": "Potongan tahu yang disajikan dengan kuah kacang dan lontong, ditambah dengan bawang goreng.",
    "Mie Rebus": "Mie yang direbus dengan kuah bening, disajikan dengan telur, sayuran, dan daging/ayam/udang.",
    "Sambal Matah": "Sambal khas Bali dengan bawang merah, cabe rawit, terasi, dan jeruk nipis.",
    "Laksa": "Mie dengan kuah santan kental dan bumbu rempah, disajikan dengan telur, ayam/udang, dan tauge.",
    "Ayam Rica-Rica": "Potongan ayam yang dimasak pedas dengan bumbu rica-rica, disajikan dengan nasi dan sayuran.",
    "Nasi Liwet": "Nasi yang dimasak dengan santan dan rempah, disajikan dengan lauk ayam/udang dan sambal.",
    "Nasi Timbel": "Nasi yang dibungkus daun pisang, disajikan dengan lauk ayam/ikan, lalapan, dan sambal.",
    "Sop Kambing": "Sup kambing dengan kuah bening yang kaya rempah, disajikan dengan nasi dan irisan daun bawang.",
    "Pindang": "Daging/ikan yang dimasak dalam kuah asam Jawa, disajikan dengan nasi dan irisan daun bawang.",
    "Ayam Betutu": "Potongan ayam yang dimasak dengan bumbu betutu khas Bali, disajikan dengan nasi dan sayuran.",
    "Bebek Betutu": "Potongan bebek yang dimasak dengan bumbu betutu khas Bali, disajikan dengan nasi dan sayuran.",
    "Sate Padang": "Potongan daging/ayam yang dipanggang dengan bumbu kacang khas Padang, disajikan dengan ketupat.",
    "Nasi Goreng Jawa": "Nasi yang digoreng dengan bumbu khas Jawa, disajikan dengan telur, ayam, dan kerupuk.",
    "Nasi Goreng Seafood": "Nasi yang digoreng dengan bumbu khas dan campuran seafood seperti cumi, udang, dan kerang.",
    "Coto Makassar": "Sup daging sapi yang kaya rempah dengan kuah yang kental, disajikan dengan ketupat.",
    "Bubur Manado": "Bubur nasi dengan berbagai lauk seperti ikan asin, telur, dan bawang goreng.",
    "Rujak Cingur": "Rujak dengan campuran buah, sayuran, dan cingur (hidung sapi) dengan saus petis.",
    "Sate Madura": "Potongan daging/ayam yang dipanggang dengan bumbu kacang khas Madura, disajikan dengan ketupat.",
    "Sate Lilit": "Sate yang dibuat dengan cara melilitkan daging/ayam pada tusuk bambu, disajikan dengan sambal.",
    "Nasi Bakar": "Nasi yang dibungkus daun pisang dan dipanggang, disajikan dengan lauk ayam/ikan, lalapan, dan sambal."
}

# Deskripsi untuk masing-masing menu minuman
deskripsi_minuman = {
    "Es Teh": "Minuman teh dingin dengan tambahan es batu dan gula.",
    "Es Jeruk": "Minuman jeruk segar dengan tambahan es batu dan gula.",
    "Jus Alpukat": "Minuman jus alpukat segar yang dicampur dengan susu kental manis.",
    "Es Cincau": "Minuman es dengan tambahan cincau dan sirup gula.",
    "Jus Mangga": "Minuman jus mangga segar yang dicampur dengan air dan gula.",
    "Es Campur": "Minuman es campur dengan tambahan berbagai buah, agar-agar, dan sirup.",
    "Es Doger": "Minuman es campur dengan tambahan kelapa muda, kolang-kaling, dan sirup.",
    "Jus Apel": "Minuman jus apel segar tanpa tambahan gula.",
    "Jus Melon": "Minuman jus melon segar yang dicampur dengan air dan gula.",
    "Jus Nanas": "Minuman jus nanas segar yang dicampur dengan air dan gula.",
    "Es Kopyor": "Minuman es dengan tambahan daging kelapa muda dan santan.",
    "Es Buah": "Minuman es dengan tambahan potongan buah segar dan sirup.",
    "Es Kelapa Muda": "Minuman es dengan tambahan potongan kelapa muda dan air gula.",
    "Jus Markisa": "Minuman jus markisa segar tanpa tambahan gula.",
    "Es Timun Suri": "Minuman es dengan tambahan potongan timun suri dan sirup.",
    "Kopi Tubruk": "Minuman kopi dengan teknik penyeduhan kopi langsung dalam cangkir.",
    "Kopi Susu": "Minuman kopi dengan tambahan susu kental manis.",
    "Kopi Hitam": "Minuman kopi tanpa tambahan susu atau gula.",
    "Teh Tarik": "Minuman teh yang disajikan dengan cara tarik agar menghasilkan busa khas.",
    "Teh Hangat": "Minuman teh hangat tanpa tambahan gula.",
    "Teh Poci": "Minuman teh yang disajikan dalam poci khas tradisional.",
    "Teh Manis": "Minuman teh dengan tambahan gula.",
    "Teh Botol": "Minuman teh kemasan botol dengan tambahan gula.",
    "Bandrek": "Minuman tradisional dengan bahan utama jahe, gula aren, dan santan.",
    "Susu Jahe": "Minuman susu dengan tambahan ekstrak jahe.",
    "Susu Kurma": "Minuman susu dengan tambahan ekstrak kurma.",
    "Soda Gembira": "Minuman soda dengan tambahan sirup dan es batu.",
    "Cendol": "Minuman tradisional dengan campuran tepung beras, santan, dan gula merah.",
    "Wedang Jahe": "Minuman tradisional dengan bahan utama jahe, gula aren, dan rempah lainnya.",
    "Wedang Ronde": "Minuman tradisional dengan bola-bola ketan dan kuah jahe.",
    "Jus Wortel": "Minuman jus wortel segar tanpa tambahan gula.",
    "Es Krim": "Minuman es krim dengan berbagai varian rasa.",
    "Air Kelapa": "Minuman segar dari air kelapa yang masih muda.",
    "Sirup Marjan": "Minuman sirup dengan berbagai varian rasa dan tambahan air.",
    "Sari Asem": "Minuman tradisional dengan campuran buah asam, gula, dan air.",
    "Bajigur": "Minuman tradisional dengan bahan utama santan, gula merah, dan rempah.",
    "Wedang Uwuh": "Minuman tradisional dengan campuran berbagai rempah seperti jahe, cengkeh, dan kayu manis.",
    "Jahe Wangi": "Minuman jahe dengan tambahan bahan utama jahe dan gula.",
    "Teh Talua": "Minuman teh dengan tambahan telur ayam.",
    "Susu Jahe Merah": "Minuman susu dengan tambahan ekstrak jahe merah.",
    "Wedang Secang": "Minuman tradisional dengan bahan utama daun secang, gula aren, dan rempah.",
    "Teh Susu": "Minuman teh dengan tambahan susu.",
    "Es Cincau Hitam": "Minuman es dengan tambahan cincau hitam dan sirup.",
    "Sari Kurma": "Minuman sari kurma segar tanpa tambahan gula.",
    "Jus Pepaya": "Minuman jus pepaya segar tanpa tambahan gula.",
    "Kolang-Kaling": "Minuman tradisional dengan tambahan kolang-kaling dan es batu.",
    "Es Oyen": "Minuman es dengan tambahan potongan oyen (buah pinang) dan gula merah.",
    "Susu Kacang Hijau": "Minuman susu dengan tambahan ekstrak kacang hijau.",
    "Es Cendol": "Minuman es dengan tambahan cendol dan sirup gula.",
    "Air Zam-Zam": "Minuman air Zam-Zam, air suci dari sumber mata air Zam-Zam di Mekkah.",
    "Susu Kurma Muda": "Minuman susu dengan tambahan ekstrak kurma muda.",
    "Jamu Beras Kencur": "Minuman jamu dengan bahan utama beras kencur dan rempah-rempah."
}
# Array bahan unik untuk makanan
bahan_makanan = [
    "Nasi", "Ayam", "Mie", "Telur", "Daging Sapi", "Santan", "Bawang Goreng", "Bawang Putih",
    "Bawang Merah", "Sosis", "Kacang Panjang", "Tauge", "Bayam", "Ketimun", "Tempe", "Tahu",
    "Kecap", "Kunyit", "Telur Ayam", "Lontong", "Daging Ayam", "Garam", "Merica", "Jahe", "Nangka Muda",
    "Lada", "Daun Jeruk", "Daun Salam", "Cabe Rawit", "Cabe Merah", "Cabe Hijau", "Kacang Tanah",
    "Kecambah", "Kol", "Wortel", "Melon", "Nanas", "Markisa", "Apel", "Kelapa Muda", "Timun Suri",
    "Kopi", "Teh", "Air Kelapa", "Sirup Gula", "Sari Kurma", "Susu", "Gula Aren", "Gula Merah",
    "Cincau", "Beras", "Kacang Hijau", "Cengkeh", "Kayu Manis", "Telur Bebek", "Sambal", "Kecap Manis",
    "Kacang Kedelai", "Minyak Goreng", "Kismis", "Jagung", "Sawi", "Saus Tomat", "Saus Sambal",
    "Saus Tiram", "Kencur", "Kecombrang", "Kluwek", "Belimbing", "Buah Naga", "Durian", "Kedondong",
    "Labu Siam", "Leci", "Nangka", "Pepaya", "Sirsak", "Bengkuang", "Asam Jawa", "Kunyit Asam",
    "Ketupat", "Bihun", "Kerupuk", "Kerang", "Cumi", "Udang", "Ikan", "Daun Singkong", "Daun Papaya",
    "Daun Pepaya", "Daun Bayam", "Daun Ketumbar", "Daun Kemangi", "Daun Pandan", "Daun Kari", "Daun Sereh",
    "Daun Terong", "Daun Kunyit", "Daun Serai", "Daun Singkong", "Daun Sambung Nyawa", "Daun Cengkeh",
    "Daun Jati", "Daun Miana", "Daun Subang", "Daun Suruhan", "Daun Talas", "Daun Tarum", "Daun Turi",
    "Daun Ungu", "Daun Waru", "Daun Gadung", "Daun Jombang", "Daun Katuk", "Daun Lembayung", "Daun Lempuyang",
    "Daun Mimba", "Daun Paitan", "Daun Cocor Bebek", "Daun Kencur", "Daun Katuk", "Daun Blimbing Wuluh",
    "Daun Kelor", "Daun Daun", "Daun Kecipir", "Daun Melinjo", "Daun Pakis", "Daun Rambutan", "Daun Sirih",
    "Daun Sonchus Arvensis", "Daun Subang", "Daun Waluh", "Daun Nangka", "Daun Brotowali", "Daun Kesum",
    "Daun Putri Malu", "Daun Gandarusa", "Daun Binahong", "Daun Landep", "Daun Bandotan", "Daun Pohpohan",
    "Daun Sambiloto", "Daun Tapak Liman", "Daun Wijayakusuma", "Daun Wungu", "Daun Gata", "Daun Rumput Fatimah",
    "Daun Pinang", "Daun Pacar Air", "Daun Ki Hujan", "Daun Kates", "Daun Kenikir", "Daun Legundi", "Daun Jarak",
    "Daun Srikaya", "Daun Sidaguri", "Daun Kayu Manis", "Daun Buah Pepaya", "Daun Asam Jawa", "Daun Blimbing Wuluh",
    "Daun Semambu", "Daun Pegagan", "Daun Sirih", "Daun Kencur", "Daun Dewa", "Daun Mangkokan", "Daun Jambu Biji",
    "Daun Cabe", "Daun Terong", "Daun Pisang", "Daun Gedi", "Daun Tomat", "Daun Sirih Merah", "Daun Kaki Kuda",
    "Daun Temurui", "Daun Kumis Kucing", "Daun Sidaguri", "Daun Alang Alang", "Daun Salam", "Daun Lemongrass",
    "Daun Mint", "Daun Lemon", "Daun Jeruk", "Daun Sirih", "Daun Kemangi", "Daun Beluntas", "Daun Meniran",
    "Daun Sirih Merah", "Daun Temurui", "Daun Kumis Kucing", "Daun Srikaya", "Daun Jati", "Daun Kayu Manis",
    "Daun Buah Pepaya", "Daun Jahe", "Daun Kates", "Daun Blimbing Wuluh", "Daun Paitan", "Daun Dewa",
    "Daun Pegagan", "Daun Asam Jawa", "Daun Ketumbar", "Daun Serai", "Daun Mint", "Daun Lemon", "Daun Pandan"
]

# Array bahan unik untuk minuman dengan tambahan buah-buahan umum
bahan_minuman = [
    "Air", "Kopi", "Teh", "Air Kelapa", "Sirup Gula", "Sari Kurma", "Susu", "Gula Aren",
    "Gula Merah", "Cincau", "Kecap Manis", "Minyak Goreng", "Kismis", "Jagung", "Jeruk Nipis",
    "Jeruk Manis", "Jeruk Bali", "Jeruk Limau", "Soda", "Sirup Buah", "Sirup Vanila", "Sirup Coklat",
    "Sirup Stroberi", "Sirup Lemon", "Sirup Mocha", "Sirup Matcha", "Sirup Tarik", "Sirup Mangga",
    "Sirup Jeruk", "Sirup Teh", "Sirup Sirsak", "Sirup Melon", "Sirup Markisa", "Sirup Blueberry",
    "Sirup Raspberry", "Sirup Anggur", "Sirup Kurma", "Sirup Apel", "Sirup Kiwi", "Sirup Leci",
    "Sirup Nangka", "Sirup Nata De Coco", "Sirup Teh Hijau", "Susu Kedelai", "Susu Almond",
    "Anggur", "Apel", "Jeruk", "Nanas", "Mangga", "Strawberry", "Blueberry", "Raspberry",
    "Kurma", "Kiwi", "Leci", "Nangka", "Melon", "Markisa", "Pisang",
]

fake = Faker('id_ID')

keyword_bahan = [
    "Bahan", "Sehat", "Makanan", "Lengkap", "Toko", "Sederhana", "Asli", "Terbaik",
    "Ramah", "Praktis", "Organik", "Modern", "Segar", "Alami", "Nikmat", "Enak",
    "Tradisional", "Utama", "Bahagia", "Khas", "Sejati", "Berkah", "Lezat", "Prima",
    "Rasa", "Bergizi", "Murni", "Sejahtera", "Harapan", "Harian", "Pokok", "Indah",
    "Favorit", "Manis", "Ceria", "Terpilih", "Makmur", "Alam", "Pulau", "Hijau",
    "Lestari", "Harmoni", "Bumi", "Sumber", "Mulia", "Lautan", "Nusantara", "Gunung",
    "Pohon", "Lingkungan", "Indonesia", "Air", "Pantai", "Hutan", "Waktu", "Tumbuh",
    "Sejahtera", "Matahari", "Cerah", "Taman", "Nusantara", "Sungai", "Tumbuhan", "Damai",
    "Buah", "Sayur", "Bumbu", "Bersih", "Sejati", "Tradisional", "Khas", "Berkah",
    "Lezat", "Praktis", "Ceria", "Pilihan", "Prima", "Sehat", "Nikmat", "Bersih"
]

def generate_indonesian_company_name():
    keyword_bahan = [ "Bahan", "Sehat", "Makanan", "Lengkap", "Toko", "Sederhana", "Asli", "Terbaik",    "Ramah", "Praktis", "Organik", "Modern", "Segar", "Alami", "Nikmat", "Enak",    "Tradisional", "Utama", "Bahagia", "Khas", "Sejati", "Berkah", "Lezat", "Prima",    "Rasa", "Bergizi", "Murni", "Sejahtera", "Harapan", "Harian", "Pokok", "Indah",    "Favorit", "Manis", "Ceria", "Terpilih", "Makmur", "Alam", "Pulau", "Hijau",    "Lestari", "Harmoni", "Bumi", "Sumber", "Mulia", "Lautan", "Nusantara", "Gunung",    "Pohon", "Lingkungan", "Indonesia", "Air", "Pantai", "Hutan", "Waktu", "Tumbuh",    "Sejahtera", "Matahari", "Cerah", "Taman", "Nusantara", "Sungai", "Tumbuhan", "Damai",    "Buah", "Sayur", "Bumbu", "Bersih", "Sejati", "Tradisional", "Khas", "Berkah",    "Lezat", "Praktis", "Ceria", "Pilihan", "Prima", "Sehat", "Nikmat", "Bersih"]

    # Add Indonesian-specific terms or words to the company name
    indonesian_keywords = [
    'Bersama', 'Maju', 'Jaya', 'Utama', 'Sejahtera', 'Mandiri', 'Nusantara', 
    'Gemilang', 'Berkah', 'Citra', 'Mitra', 'Murni', 'Harapan', 'Sukses', 
    'Cemerlang', 'Abadi', 'Prima', 'Mega', 'Inti', 'Sentosa', 'Pertama',
    'Lestari', 'Mandala', 'Eka', 'Surya', 'Bhakti', 'Karya', 'Kencana', 
    'Harapan', 'Makmur', 'Wijaya', 'Bakti', 'Nusa', 'Dharma', 'Maju',
    'Purnama', 'Jasatama', 'Sinar', 'Kusuma', 'Harmoni', 'Jayakarta', 
    'Pusaka', 'Dian', 'Angkasa', 'Anugerah', 'Mulia', 'Pratama', 'Raya',
    'Agung', 'Wahana', 'Mitra', 'Global', 'Jasa', 'Kreatif', 'Karya', 
    'Energi', 'Sejahtera', 'Konsultan', 'Bersinar', 'Nusa', 'Tata', 'Prima',
    'Bangun', 'Konsultama', 'Prestasi', 'Karya', 'Intan', 'Prakarsa', 
    'Megah', 'Cipta', 'Sakti', 'Utama', 'Wira', 'Maju', 'Kencana', 'Cendekia',
    'Mandiri', 'Maju', 'Sejahtera', 'Mitra', 'Wijaya', 'Mitra', 'Maju', 
    'Lestari', 'Bersama', 'Mitra', 'Maju', 'Nusantara', 'Maju', 'Surya',
    'Berkat', 'Bersama', 'Maju', 'Jaya', 'Maju', 'Makmur', 'Sentosa', 
    'Sejahtera', 'Sejahtera', 'Karya', 'Harmoni', 'Mitra', 'Sejahtera', 
    'Nusantara', 'Jaya', 'Karya', 'Utama', 'Mandiri', 'Citra', 'Sentosa', 
    'Jasa', 'Maju', 'Mitra', 'Prima', 'Mega', 'Sentosa', 'Utama', 'Gemilang',
    'Jaya', 'Maju', 'Nusa', 'Jaya', 'Surya', 'Harapan', 'Sentosa', 'Wijaya',
    'Bhakti', 'Karya', 'Utama', 'Makmur', 'Dharma', 'Maju', 'Purnama', 'Jasa',
    'Mitra', 'Pertama', 'Sentosa', 'Maju', 'Mega', 'Utama', 'Bersama', 'Maju',
    'Utama', 'Sejahtera', 'Mandiri', 'Nusantara', 'Gemilang', 'Berkah', 
    'Citra', 'Mitra', 'Murni', 'Harapan', 'Sukses', 'Cemerlang', 'Abadi', 
    'Prima', 'Mega', 'Inti', 'Sentosa', 'Pertama', 'Lestari', 'Mandala', 
    'Eka', 'Surya', 'Bhakti', 'Karya', 'Kencana', 'Harapan', 'Makmur', 
    'Wijaya', 'Bakti', 'Nusa', 'Dharma', 'Maju', 'Purnama', 'Jasatama', 
    'Sinar', 'Kusuma', 'Harmoni', 'Jayakarta', 'Pusaka', 'Dian', 'Angkasa', 
    'Anugerah', 'Mulia', 'Pratama', 'Raya', 'Agung', 'Wahana', 'Mitra', 
    'Global', 'Jasa', 'Kreatif', 'Karya', 'Energi', 'Sejahtera', 'Konsultan', 
    'Bersinar', 'Nusa', 'Tata', 'Prima', 'Bangun', 'Konsultama', 'Prestasi', 
    'Karya', 'Intan', 'Prakarsa', 'Megah', 'Cipta', 'Sakti', 'Utama', 'Wira', 
    'Maju', 'Kencana', 'Cendekia', 'Mandiri', 'Maju', 'Sejahtera', 'Mitra', 
    'Wijaya', 'Mitra', 'Maju', 'Lestari', 'Bersama', 'Mitra', 'Maju', 'Nusantara', 
    'Maju', 'Surya', 'Berkat', 'Bersama', 'Maju', 'Jaya', 'Maju', 'Makmur', 
    'Sentosa', 'Sejahtera', 'Sejahtera', 'Karya', 'Harmoni', 'Mitra', 'Sejahtera', 
    'Nusantara', 'Jaya', 'Karya', 'Utama', 'Mandiri', 'Citra', 'Sentosa', 'Jasa', 
    'Maju', 'Mitra', 'Prima', 'Mega', 'Sentosa', 'Utama', 'Gemilang', 'Jaya', 
    'Maju', 'Nusa', 'Jaya', 'Surya', 'Harapan', 'Sentosa', 'Wijaya', 'Bhakti', 
    'Karya', 'Utama', 'Makmur', 'Dharma', 'Maju', 'Purnama', 'Jasa', 'Mitra', 
    'Pertama', 'Sentosa', 'Maju', 'Mega', 'Utama'
]

    nama_perusahaan_orang = [
    "Budi", "Santoso", "Agung", "Wijaya", "Putra", "Jaya", "Utama", "Mulya", "Sari",
    "Hadi", "Pratama", "Susanto", "Kurniawan", "Gunawan", "Susilo", "Wibowo", "Yanto",
    "Widodo", "Dewi", "Lestari", "Agus", "Hermawan", "Hidayat", "Nugroho", "Saputra",
    "Kusuma", "Surya", "Putri", "Dewanto", "Purnomo", "Sulistyo", "Saputro", "Setiawan",
    "Permadi", "Prasetyo", "Sukardi", "Bambang", "Yulianto", "Kristanto", "Agustina",
    "Ningsih", "Rahayu", "Kusumawati", "Setiawati", "Susanti", "Rahmawati", "Suryani",
    "Fitriani", "Indah", "Wahyuni", "Saputri", "Wulandari", "Septiani", "Puspitasari",
    "Utami", "Nur", "Laila", "Handayani", "Hasanah", "Nurhayati", "Putri", "Kusumo",
    "Wibisono", "Supriyanto", "Yusuf", "Indrawan", "Setiadi", "Wahyudin", "Suryadi",
    "Wicaksono", "Suharto", "Purnomo", "Suryanto", "Saputro"
]

    status_company = ["UD", "CV", "PT", "Firma"]

    indonesian_company_name = fake.random_element(status_company) + " " + fake.random_element(nama_perusahaan_orang) + " " + fake.random_element(keyword_bahan) + " " +  fake.random_element(indonesian_keywords)

    return indonesian_company_name

email_extensions = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "mail.com", "protonmail.com", "icloud.com", "zoho.com", "aol.com", "yandex.com"
                    "inbox.com", "gmx.com", "fastmail.com", "hushmail.com", "tutanota.com", "lavabit.com", "keemail.me", "mailfence.com", "riseup.net", "scryptmail.com",
                    "tutanota.com", "countermail.com", "ctemplar.com", "mailbox.org", "posteo.de", "runbox.com", "startmail.com", "mailbox.org", "posteo.de", "runbox.com",
                    "startmail.com", "tutanota.com", "countermail.com", "ctemplar.com", "mailbox.org", "posteo.de", "runbox.com", "startmail.com", "mailbox.org", "posteo.de",
                    "runbox.com", "startmail.com", "tutanota.com", "countermail.com", "ctemplar.com", "mailbox.org", "posteo.de", "runbox.com", "startmail.com", "mailbox.org",
                    "posteo.de", "runbox.com", "startmail.com", "tutanota.com", "countermail.com", "ctemplar.com", "mailbox.org", "posteo.de", "runbox.com", "startmail.com",
                    "mailbox.org", "posteo.de", "runbox.com", "startmail.com", "tutanota.com", "countermail.com", "ctemplar.com", "mailbox.org", "posteo.de", "runbox.com",
                    "startmail.com", "mailbox.org", "posteo.de", "runbox.com", "startmail.com", "tutanota.com", "countermail.com", "ctemplar.com", "mailbox.org", "posteo.de",
                    ]

def generate_email(name, email_extensions):
    # Remove punctuation from the name
    name = name.translate(str.maketrans('', '', string.punctuation))
    # Split the name into words and take the first two words
    name_words = name.split()[:2]
    name = "".join(name_words).lower()
    
    # Randomly select whether to include a random number based on probability
    include_random_number = random.choice([True, False])
    # Generate a random number if include_random_number is True
    random_number = ''
    if include_random_number:
        random_number = str(random.randint(1, 100))
    
    # Randomly select an email extension
    extension = random.choice(email_extensions)
    
    # Concatenate name, random number, and email extension to form the email
    email = f"{name}{random_number}@{extension}"
    return email

def generate_nomor_telepon():
    operator_prefix_numbers = [
        "857", "858", "859", "812", "813", "814", "815", "816", "855", "856",  # Indosat
        "877", "878", "879", "813", "814", "815", "817", "818", "819",        # XL
        "881", "882", "883", "884", "885",                                    # Smartfren
        "895", "896", "897", "898", "899",                                    # Tri
        "821", "822", "823", "824", "825", "826", "827", "828", "829",        # Axis
        "831", "832", "833", "834", "835", "836", "837", "838", "839",        # Bolt
        "841", "842", "843", "844", "845", "846", "847", "848", "849",        # By.U
        "865", "866", "867", "868", "869"                                     # Indosat Ooredoo
        "891", "892", "893", "894", "895", "896", "897", "898", "899"         # Smartfren
        "821", "822", "823", "824", "825", "826", "827", "828", "829"         # Axis
    ]
    # Randomly select an operator prefix number
    random_operator_prefix_number = random.choice(operator_prefix_numbers)
    # Calculate the maximum length for remaining digits based on operator prefix length and the constraint
    max_remaining_digits_length = min(13 - len("+62" + random_operator_prefix_number), random.randint(5, 7))
    # Generate the remaining digits
    remaining_digits = ''.join(random.choices('0123456789', k=max_remaining_digits_length))
    # Combine the country code, operator prefix number, and remaining digits
    return "+62" + random_operator_prefix_number + remaining_digits


def is_valid_email(email):
    # Pola regex untuk memeriksa validitas email
    pattern = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}$'
    
    # Memeriksa apakah email sesuai dengan pola regex
    if re.match(pattern, email):
        return True
    else:
        return False
    
# Must unique entity
unique_telephone = set()
unique_email = set()

def generate_komentar():
    tipe_komentar = ["Tempat", "Menu", "Kebersihan", "Pelayanan"]
    sifat_tempat = [
        "sangat tidak nyaman", "tidak nyaman", "cukup nyaman", "lumayan nyaman", "nyaman",
        "sangat nyaman", "terlalu bising", "kurang tenang", "tenang", "sepi", "terlalu sepi",
        "ramai", "terlalu ramai", "sempit", "cukup luas", "luas", "lega", "terlalu gelap",
        "terang", "terlalu terang", "bersih", "sangat bersih", "kotor", "cukup bersih",
        "tidak bersih", "indah", "jelek"
    ]
    sifat_menu = [
        "sangat tidak enak", "tidak enak", "cukup enak", "lumayan enak", "enak", "sangat enak",
        "terlalu manis", "kurang manis", "manis", "pahit", "terlalu pahit", "asin", "terlalu asin",
        "gurih", "segar", "tidak segar", "hambar", "pedas", "terlalu pedas", "kurang pedas",
        "lezat", "nikmat", "tidak lezat"
    ]
    sifat_kebersihan = [
        "sangat kotor", "kotor", "cukup bersih", "lumayan bersih", "bersih", "sangat bersih",
        "jorok", "bau", "wangi", "tidak wangi", "berdebu", "tidak berdebu", "rapih", "acak-acakan"
    ]
    sifat_pelayanan = [
        "sangat buruk", "buruk", "cukup baik", "lumayan baik", "baik", "sangat baik",
        "ramah", "tidak ramah", "sopan", "tidak sopan", "cepat", "lambat", "kurang cepat",
        "terlalu cepat", "profesional", "tidak profesional", "menyenangkan", "tidak menyenangkan"
    ]

    random_tipe_komentar = fake.random_element(tipe_komentar)
    if random_tipe_komentar == "Tempat":
        random_sifat = fake.random_element(sifat_tempat)
        return f"Tempat {random_sifat}."
    elif random_tipe_komentar == "Menu":
        random_sifat = fake.random_element(sifat_menu)
        return f"Menu {random_sifat}."
    elif random_tipe_komentar == "Kebersihan":
        random_sifat = fake.random_element(sifat_kebersihan)
        return f"Kebersihan {random_sifat}."
    elif random_tipe_komentar == "Pelayanan":
        random_sifat = fake.random_element(sifat_pelayanan)
        return f"Pelayanan {random_sifat}."
    
# Generate dummy data for Menu table
# Constraint make sure that the same menu can't be added twice
menu_data = []
existing_menus = set()
for name in (menu_makanan+menu_minuman):
    id_menu = str(uuid.uuid4())
    nama = name
    harga = random.randint(500, 10000) * 100 if nama in menu_makanan else random.randint(301, 500) * 100
    rating = random.randint(0, 5)
    deskripsi = deskripsi_makanan[nama] if nama in menu_makanan else deskripsi_minuman[nama]
    tipe = "Makanan" if nama in menu_makanan else "Minuman"
    if (id_menu, nama, harga, rating, deskripsi, tipe) not in existing_menus:
        menu_data.append((id_menu, nama, harga, rating, deskripsi, tipe))
        existing_menus.add((id_menu, nama, harga, rating, deskripsi, tipe))
   
print("Menu data: DONE")
    
# Generate dummy data for PenyediaBahan table
# Constraint make sure that the same bahan can't be provided by the same penyedia and nomor_telepon must be unique also email
penyedia_data = []
for _ in range(500):
    while True:
        nama = generate_indonesian_company_name()
        email = generate_email(nama, email_extensions)
        nomorTelepon = generate_nomor_telepon()
        
        if not any(existing_nama == nama for _, existing_nama, _ in penyedia_data) and nomorTelepon not in unique_telephone and email not in unique_email:
            penyedia_data.append((email, nama, nomorTelepon))
            unique_telephone.add(nomorTelepon)
            unique_email.add(email)
            break

print("Penyedia data: DONE")  

# Generate dummy data for Bahan table
# Constraint make sure that the same bahan can't be added twice
bahan_data = []
for name in ((bahan_makanan+bahan_minuman)):
        id_bahan = str(uuid.uuid4())
        nama = name
        stok = random.randint(1, 200)  # Random stock between 1 and 1000
        
        if not any(existing_id == id_bahan or existing_nama == nama for existing_id, existing_nama, _ in bahan_data):
            bahan_data.append((id_bahan, nama, stok))
print("Bahan data: DONE")

# Generate dummy data for Pegawai table
# Constraint make sure that the same nik_pegawai and nama can't be added twice and nomor_telepon must be unique
pegawai_data = []
for _ in range(50):
    while True:
        nik_pegawai = fake.random_int(min=1000000000000000, max=9999999999999999)  # Random 16-digit number
        nama = fake.name()
        # Validatiion for name
        if "dr." in nama or "S." in nama or "M." in nama:
            continue
        alamat = fake.address()
        nomorTelepon = generate_nomor_telepon()
        if(not any(existing_nik == nik_pegawai for existing_nik, _, _, _ in pegawai_data) and not any(existing_nama == nama for _, existing_nama, _, _ in pegawai_data) and nomorTelepon not in unique_telephone):
            pegawai_data.append((nik_pegawai, nama, alamat, nomorTelepon))
            unique_telephone.add(nomorTelepon)
            break
print("Pegawai data: DONE")

# Generate dummy data for Pengunjung table
# Constraint make sure that the same email and nama can't be added twice and nomor_telepon must be unique also email
pengunjung_data = []
for _ in range(2000):
    while True:
        nama = fake.name()
        email = generate_email(nama, email_extensions)
        nomorTelepon = generate_nomor_telepon()
       
        if not any(existing_nama == nama for _, existing_nama, _ in pengunjung_data) and nomorTelepon not in unique_telephone and email not in unique_email:
            pengunjung_data.append((email, nama, nomorTelepon))
            unique_telephone.add(nomorTelepon)
            unique_email.add(email) 
            break
        
print("Pengunjung data: DONE")
  
          
#  Generate dummy data for Transaksi table
# Constraint make sure that the same nomor_transaksi and email_pengunjung can't be added twice and email must be unique also email
# Many to many by randomize the email pengunjung and nik
transaksi_data = []
for _ in range(3000):
    while True:
        nomor_transaksi = str(uuid.uuid4())
        metode_pembayaran = random.choice(['Tunai', 'Kartu Kredit', 'QRIS', 'Debit'])
        total_harga = random.randint(100, 100000) * 100
        date = fake.date_time_between(start_date="-3y", end_date="now", tzinfo=None)
        email_pengunjung = random.choice(pengunjung_data)[0]
        nik_pegawai = random.choice(pegawai_data)[0]
        
        if not any(existing_nomor_transaksi == nomor_transaksi for existing_nomor_transaksi, _, _, _, _, _ in transaksi_data):
            transaksi_data.append((nomor_transaksi, metode_pembayaran, total_harga, date, email_pengunjung, nik_pegawai))
            break
    
print("Transaksi data: DONE")

# RELATIONSHIP SECTION
# Generate dummy data for Makanan table
makanan_data = []
for menu in menu_data:
    id_menu = menu[0]
    if menu[5] == 'Makanan':  # Pastikan nama menu sesuai dengan tipe menu
        while True:
            tingkat_kepedasan = random.randint(0, 5)
            if not any(existing_id_menu == id_menu for existing_id_menu, _ in makanan_data):
                makanan_data.append((id_menu, tingkat_kepedasan))
                break
print("Makanan data: DONE")

# Generate dummy data for Minuman table
minuman_data = []
for menu in menu_data:
    id_menu = menu[0]
    if menu[5] == 'Minuman':  # Pastikan nama menu sesuai dengan tipe menu
        while True:
            is_dingin = random.choice([True, False])
            if not any(existing_id_menu == id_menu for existing_id_menu, _ in minuman_data):
                minuman_data.append((id_menu, is_dingin))
                break
print("Minuman data: DONE")

# Generate dummy data for BahanMenu table
bahan_menu_data = []
for menu in menu_data:
    id_menu = menu[0]
    jumlah = random.randint(0, 10)
    random_count = random.randint(1, 10)
    while random_count > 0:
        random_count = random.randint(1, 10)
        # Make sure menu bahan is from correct makanan or minuman
        if menu[4] == 'Makanan':
            bahan = random.choice(bahan_makanan)
        else:
            bahan = random.choice(bahan_minuman)
        id_bahan = None
        for bd in bahan_data:
            if bd[1] == bahan:
                id_bahan = bd[0]
        if id_bahan is not None and not any(existing_id_menu == id_menu and existing_id_bahan == id_bahan for existing_id_menu, existing_id_bahan, _ in bahan_menu_data):
            bahan_menu_data.append((id_menu, id_bahan, jumlah))
            random_count -= 1

print("Bahan Menu data: DONE")

     
# Generate dummy data for PembelianBahan table
pembelian_bahan_data = []
existings = {bahan[0]: 0 for bahan in bahan_data}  # Track existing stock

for penyedia in penyedia_data:
    email_penyedia = penyedia[0]
    random_count = random.randint(1, 10)
    while random_count > 0:
        id_bahan = random.choice(bahan_data)[0]
        harga = random.randint(1, 3) * 100
        jumlah_pembelian = random.randint(0, 20)
        tanggal_pembelian = fake.date_time_between(start_date="-3y", end_date="now", tzinfo=None)
        
        if existings[id_bahan] + jumlah_pembelian <= 200:
            if not any(existing_email_penyedia == email_penyedia and existing_id_bahan == id_bahan for existing_email_penyedia, existing_id_bahan, _, _, _ in pembelian_bahan_data):
                pembelian_bahan_data.append((email_penyedia, id_bahan, harga, jumlah_pembelian, tanggal_pembelian))
                existings[id_bahan] += jumlah_pembelian
                random_count -= 1
        else:
            random_count -= 1

print("Pembelian Bahan data: DONE")

# Update stock in bahan_data based on existings
for i, bahan in enumerate(bahan_data):
    if bahan[0] in existings:
        bahan_data[i] = (bahan[0], bahan[1], existings[bahan[0]])


print("Update Pembelian Bahan data: DONE")

# Generate dummy data for DetailTransaksi table
detail_transaksi_data = []
for transaksi in transaksi_data:
    nomor_transaksi = transaksi[0]
    random_count = random.randint(1, 10)
    while random_count > 0:
        id_detail_transaksi = str(uuid.uuid4())
        kuantitas = random.randint(1, 10)  # Updated constraint
        id_menu = random.choice(menu_data)[0]
        if(not any(existing_nomor_transaksi == nomor_transaksi and existing_id_menu == id_menu and id_detail_transaksi == idd for existing_nomor_transaksi, idd, _, existing_id_menu in detail_transaksi_data)):
            detail_transaksi_data.append((nomor_transaksi, id_detail_transaksi, kuantitas, id_menu))
            random_count -= 1
            
print("Detail Transaksi data: DONE")
            
# Update transaksi based on detail_transaksi
for transaksi in transaksi_data:
    nomor_transaksi = transaksi[0]
    metode_pembayaran = transaksi[1]
    total_harga = 0
    for detail_transaksi in detail_transaksi_data:
        if detail_transaksi[0] == nomor_transaksi:
            for menu in menu_data:
                if menu[0] == detail_transaksi[3]:
                    total_harga += menu[2] * detail_transaksi[2]
                    
    if total_harga < 20000 and metode_pembayaran == "QRIS":
        metode_pembayaran = 'Tunai'
    elif total_harga < 50000 and metode_pembayaran == "Debit":
            metode_pembayaran = 'Tunai'
    elif total_harga < 100000 and metode_pembayaran == "Kartu Kredit":
                metode_pembayaran = 'Tunai'
    
    transaksi_data[transaksi_data.index(transaksi)] = (transaksi[0], metode_pembayaran, total_harga, transaksi[3], transaksi[4], transaksi[5])

        
print("Update Transaksi data: DONE")



# Generate dummy data for RatingMenu table
rating_menu_data = []
max_count = 2000
count = 0
for detail in detail_transaksi_data:
    id_detail_transaksi = detail[1]
    id_feedback = detail[0]
    rating = random.randint(0, 5)
    if(rating <= 4):
        if(count == max_count):
            rating = random.randint(4, 5)
        else:
            count += 1
    if(not any(existing_id_detail_transaksi == id_detail_transaksi and existing_id_feedback == id_feedback for existing_id_detail_transaksi, existing_id_feedback, _ in rating_menu_data)):
        rating_menu_data.append((id_detail_transaksi, id_feedback, rating))

print("Detail Rating Menu data: DONE")


# Generate dummy data for Feedback table
max_count = 200
count_pelayanan = 0
count_kebersihan = 0
feedback_data = []
for rating in rating_menu_data:
    id_feedback = rating[1]
    time_feedback = tanggal_stok = fake.date_time_between(start_date="-3y", end_date="now", tzinfo=None)
    rating_pelayanan = random.randint(0, 5)
    if(rating_pelayanan<4):
        count_pelayanan += 1
        if(count_pelayanan == max_count):
            rating_pelayanan = random.randint(4, 5)
    rating_kebersihan = random.randint(0, 5)
    if rating_kebersihan < 4:
        count_kebersihan += 1
        if(count_kebersihan == max_count):
            rating_kebersihan = random.randint(4, 5)
    if rating_kebersihan==0 and rating_pelayanan==5:
        rating_pelayanan-=1
    komentar = generate_komentar()
    rating_menu_overall = 0
    if(not any(existing_id_feedback == id_feedback for existing_id_feedback, _, _, _, _, _ in feedback_data)):
        feedback_data.append((id_feedback, time_feedback, rating_pelayanan, rating_kebersihan, komentar, rating_menu_overall))

print("Detail Feedback data: DONE")

# Update menu based on detail transaksi and rating menu
for menu in menu_data:
    sum_rating = 0
    count = 0
    for detail_transaksi in detail_transaksi_data:
        if detail_transaksi[3] == menu[0]:
            for rating in rating_menu_data:
                if rating[0] == detail_transaksi[1]:
                    sum_rating = sum_rating + rating[2]
                    count += 1
    avg_rating = sum_rating / count
    avg_rating = round(avg_rating, 2)
    menu_data[menu_data.index(menu)] = (menu[0], menu[1], menu[2], avg_rating, menu[4], menu[5])

print("Update Menu data: DONE")

# Update feedback based on rating menu
for feedback in feedback_data:
    sum_rating = 0
    count = 0
    for rating in rating_menu_data:
        if feedback[0] == rating[1]:
            sum_rating += rating[2]
            count += 1
    rating_overall = sum_rating / count
    rating_overall = round(rating_overall, 2)
    feedback_data[feedback_data.index(feedback)] = (feedback[0], feedback[1], feedback[2], feedback[3], feedback[4], rating_overall)

print("Update feedback data: DONE")
    

if(len(menu_data) <=1) or (len(penyedia_data) <=1) or (len(bahan_data) <=1) or (len(pegawai_data) <=1) or (len(pengunjung_data) <=1) or len(transaksi_data) <=1 or len(makanan_data) <=1 or len(minuman_data) <=1 or len(bahan_menu_data) <=1 or len(pembelian_bahan_data) <=1 or len(detail_transaksi_data) <=1:
    print("Data is empty")


    
for transaksi in transaksi_data:
    if transaksi[2] < 20000 and transaksi[1] == "QRIS":
        print("Salah qris")
    elif transaksi[2] < 50000 and transaksi[1] == "Debit":
        print("Salah debit")
    elif transaksi[2] < 100000 and transaksi[1] == "Kartu Kredit":
        print("Salah kredit")

with open('migrate.sql', 'w') as file:
   
    file.write('INSERT INTO Menu (idMenu, nama, harga, rating, deskripsi, tipe) VALUES\n')
    file.write(','.join([f"('{m[0]}', '{m[1]}', {m[2]}, {m[3]}, '{m[4]}', '{m[5]}')" for m in menu_data]) + ';\n\n')

    file.write('INSERT INTO PenyediaBahan (email, nama, nomorTelepon) VALUES\n')
    file.write(','.join([f"('{p[0]}', '{p[1]}', '{p[2]}')" for p in penyedia_data]) + ';\n\n')

    file.write('INSERT INTO Bahan (idBahan, nama, stok) VALUES\n')
    file.write(','.join([f"('{b[0]}', '{b[1]}', {b[2]})" for b in bahan_data]) + ';\n\n')

    file.write('INSERT INTO Pegawai (NIKPegawai, nama, alamat, nomorTelepon) VALUES\n')
    file.write(','.join([f"('{p[0]}', '{p[1]}', '{p[2]}', '{p[3]}')" for p in pegawai_data]) + ';\n\n')

    file.write('INSERT INTO Pengunjung (email, nama, nomorTelepon) VALUES\n')
    file.write(','.join([f"('{p[0]}', '{p[1]}', '{p[2]}')" for p in pengunjung_data]) + ';\n\n')

    file.write('INSERT INTO Transaksi (nomorTransaksi, metodePembayaran, totalHarga, tanggalTransaksi, emailPengunjung, NIKPegawai) VALUES\n')
    file.write(','.join([f"('{t[0]}', '{t[1]}', {t[2]}, '{t[3]}', '{t[4]}', '{t[5]}')" for t in transaksi_data]) + ';\n\n')

    file.write('INSERT INTO Makanan (idMenu, tingkatKepedasan) VALUES\n')
    file.write(','.join([f"('{m[0]}', {m[1]})" for m in makanan_data]) + ';\n\n')

    file.write('INSERT INTO Minuman (idMenu, isDingin) VALUES\n')
    file.write(','.join([f"('{m[0]}', {m[1]})" for m in minuman_data]) + ';\n\n')

    file.write('INSERT INTO BahanMenu (idMenu, idBahan, jumlah) VALUES\n')
    file.write(','.join([f"('{bm[0]}', '{bm[1]}', {bm[2]})" for bm in bahan_menu_data]) + ';\n\n')

    file.write('INSERT INTO PembelianBahan (emailPenyedia, idBahan, harga, jumlahPembelian, tanggalPembelian) VALUES\n')
    file.write(','.join([f"('{pb[0]}', '{pb[1]}', {pb[2]}, {pb[3]}, '{pb[4]}')" for pb in pembelian_bahan_data]) + ';\n\n')

    file.write('INSERT INTO DetailTransaksi (nomorTransaksi, idDetailTransaksi, kuantitas, idMenu) VALUES\n')
    file.write(','.join([f"('{dt[0]}', '{dt[1]}', {dt[2]}, '{dt[3]}')" for dt in detail_transaksi_data]) + ';\n\n')

    file.write('INSERT INTO Feedback (idFeedback, waktuFeedback, ratingPelayanan, ratingKebersihan, komentar, ratingMenuOverall) VALUES\n')
    file.write(','.join([f"('{fb[0]}', '{fb[1]}', {fb[2]}, {fb[3]}, '{fb[4]}', {fb[5]})" for fb in feedback_data]) + ';\n\n')

    file.write('INSERT INTO RatingMenu (idDetailTransaksi, idFeedback, rating) VALUES\n')
    file.write(','.join([f"('{rm[0]}', '{rm[1]}', {rm[2]})" for rm in rating_menu_data]) + ';\n\n')