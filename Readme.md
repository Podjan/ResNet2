# Seattle AirBnB Listing Dataset (Rakamin Academy Final Project)
Dalam final project ini, kelompok kami ingin membuat machine learning yang bertujuan untuk memberikan wawasan yang diperlukan dan dapat membantu AirBnB dalam meningkatkan jumlah customer di Kota Seattle sehingga posisi mereka di pasar Kota Seattle semakin kuat. 

## Kelompok 2 ResNet
- Project Manager : Mohammad Fauzan
- Data Analyst    : Indah Mutiah Utami. MZ
- Data Engineer   : Yusuf Nafi Farhan
- Data Scientist  : Julian

## Daftar Isi
- [Prerequisites](#prerequisites)
- [Penjelasan Sebelum EDA](#penjelasan-sebelum-eda)
- [EDA](#eda)


## Prerequisites
1. Download data [here](https://drive.google.com/drive/folders/1q0uoNhUzHYL3TmhfwtFL-Xnb26rwOzRF?usp=sharing)
2. Clone repositori ini:
   ```bash
   git clone https://github.com/Podjan/ResNet2.git

## Penjelasan Sebelum EDA
### Dataset
Ada tiga dataset pada pekerjaan kali ini. 
1. **Calendar** yang berisi tentang data penghasilan dari AirBnB selama setahun.
2. **Listings** yang berisi tentang data detail lengkap mengenai setiap listings.
3. **Reviews** yang berisi tentang ulasan dari setiap listing.

### Beberapa hal yang perlu diperhatikan
1. Penentuan kolom dan dataset yang diambil dilakukan diawal untuk meminimalisir resiko dan membuang yang tidak perlu.
2. Poin pertama bukan berarti selanjutnya tidak ada pengambilan keputusan untuk membuang kolom. Jika ada kolom yang korelasinya mendekati 1 kemungkinan akan dibuang juga.
3. Penentuan kolom dan dataset diawal juga didasari oleh goal dan objective yang ingin dicapai.

### Goal dan Objectives
#### Goal
Goals yang ingin dicapai dalam studi case ini adalah meningkatkan jumlah customer AirBnB di tahun berikutnya dengan memberikan wawasan yang akurat kepada pemilik AirBNB mengenai jenis properti yang paling diminati oleh tamu dan tingkat kepuasan para tamu, sehingga mereka dapat mengoptimalkan strategi pemasaran, penetapan harga, dan pengelolaan inventaris di tahun berikutnya.
#### Objectives
1. Memprediksi jumlah pengunjung di tahun berikutnya berdasarkan kualitas dan harga.
2. Meningkatkan tingkat kepuasan customer terhadap kualitas hotel yang pengunjungnya sepi.
3. Memprediksi harga hotel di tahun depan untuk memberikan rekomendasi harga yang tepat di setiap hotel.

### Langkah awal
1. Memisahkan kolom date menjadi kolom bulan dan tahun.
2. Melakukan groupby di dataset calendar yang bertujuan mendapatkan median price dan jumlah customer (jumlah id) dari masing listing id di setiap bulannya.
   ``` python
   dfcg = dfc_cleaned.groupby(['listing_id', 'tahun', 'bulan']).agg(
       median_price=('payment', 'median'),
       jumlah_id=('listing_id', 'size')
   ).reset_index()
   ```
3. Mengubah beberapa kolom di dataset listing menjadi tipe yang diinginkan
   ``` python
   # Mengubah data yang bertipe string ke float
   dfl2['price'] = dfl2['price'].str.replace('$', '').str.replace(',', '').astype(float)
   dfl2['weekly_price'] = dfl2['weekly_price'].str.replace('$', '').str.replace(',', '').astype(float)
   dfl2['monthly_price'] = dfl2['monthly_price'].str.replace('$', '').str.replace(',', '').astype(float)
   dfl2['host_response_rate'] = dfl2['host_response_rate'].str.replace('%', '').astype(float)
   dfl2['host_acceptance_rate'] = dfl2['host_acceptance_rate'].str.replace('%', '').astype(float)
   # Mengubah data yang beripe integer ke string
   dfl2['host_id'] = dfl2['host_id'].astype(str)
   dfl2['latitude'] = dfl2['latitude'].astype(str)
   dfl2['longitude'] = dfl2['longitude'].astype(str)
   ```
4. Memasukkan kolom jumlah id dari dataset calendar ke dataset listing.
   ```python
   dfcg2 = dfcg [['listing_id', 'jumlah_id']]
   dfcg3 = dfcg2.groupby(['listing_id']).agg(jumlah_id=('jumlah_id', 'sum')).reset_index()
   # Menggabungkan dfcg3 dan dfl2 berdasarkan 'listing_id dengan tujuan memembuat kolom jumlah_id di dataset listing
   dfl3 = pd.merge(dfcg3, dfl2, on='listing_id', how='left')
   ```
5. Sekarang kita punya dua dataframe/dataset yang akan diolah
   ```python
   df1 = dfcg.sort_values(by=['listing_id','tahun', 'bulan_num'])
   df2 = dfl3
   ```

### Kolom yang diambil
#### df1 (dataset calendar final)
- listing_id
- tahun
- bulan
- bulan_num
- median_price
- jumlah_id
#### df2 (dataset listing final)
- listing_id
- jumlah_id
- name
- host_id
- host_response_time
- host_response_rate
- host_acceptance_rate
- host_is_superhost
- host_identity_verified
- zipcode
- latitude
- longitude
- is_location_exact
- property_type
- room_type
- accommodates
- bathrooms
- bedrooms
- beds
- bed_type
- price
- weekly_price
- monthly_price
- guests_included
- minimum_nights
- maximum_nights
- review_scores_rating
- review_scores_accuracy
- review_scores_cleanliness
- review_scores_checkin
- review_scores_communication
- review_scores_location
- review_scores_value
- instant_bookable
- cancellation_policy
- require_guest_profile_picture
- require_guest_phone_verification

## EDA
### Descriptive Statistic
#### df1

#### df2

### Univariate Analysis
#### df1

#### df2

### Multivariate Analysis
#### df1
**Heatmap df1**
Heatmap ini menunjukkan matriks korelasi antar variabel dalam dataset, yaitu tahun, bulan_num, median_price, dan jumlah_id. Berikut penjelasan dari korelasi antar variabel:
1. tahun dengan bulan_num (-0.40):
Korelasi negatif lemah, yang menunjukkan bahwa ketika tahun meningkat, bulan_num cenderung sedikit menurun. Namun, hubungan ini tidak terlalu kuat, sehingga tidak ada keterkaitan yang jelas antara tahun dan bulan_num.
2. tahun dengan median_price (0.01):
Korelasi hampir nol, yang menunjukkan tidak ada hubungan linear antara tahun dan median_price. Ini berarti median_price tidak berubah secara konsisten seiring perubahan tahun.
3. tahun dengan jumlah_id (-0.76):
Korelasi negatif cukup kuat, menunjukkan bahwa ketika tahun meningkat, jumlah_id cenderung menurun. Hal ini bisa menunjukkan adanya tren menurun pada jumlah_id seiring bertambahnya tahun.
4. bulan_num dengan median_price (0.05):
Korelasi hampir nol, menunjukkan tidak ada hubungan linear yang signifikan antara bulan_num dan median_price. Ini berarti bahwa perubahan dalam bulan_num tidak mempengaruhi median_price secara konsisten.
5. bulan_num dengan jumlah_id (0.51):
Korelasi positif moderat, yang menunjukkan bahwa ketika bulan_num meningkat, jumlah_id cenderung ikut meningkat. Ada kecenderungan bahwa jumlah_id lebih tinggi di bulan-bulan tertentu.
6. median_price dengan jumlah_id (0.01):
Korelasi sangat rendah atau hampir nol, yang menunjukkan bahwa tidak ada hubungan linear antara median_price dan jumlah_id. Dengan kata lain, median_price dan jumlah_id tidak mempengaruhi satu sama lain secara langsung.

#### df2
**Heatmap df2**
1. Korelasi Antar Variabel Fasilitas Properti 
accommodates, bathrooms, bedrooms, dan beds:
Terdapat korelasi yang cukup tinggi di antara variabel ini. Contohnya:
- accommodates dan bedrooms memiliki korelasi sebesar 0.77.
- bedrooms dan beds memiliki korelasi sebesar 0.75.
Korelasi yang tinggi ini menunjukkan bahwa jumlah kamar tidur (bedrooms), kamar mandi (bathrooms), dan tempat tidur (beds) cenderung bertambah seiring bertambahnya kapasitas (accommodates) properti. Hal ini logis karena properti yang lebih besar biasanya memiliki lebih banyak fasilitas.
price, weekly_price, dan monthly_price:
- Korelasi yang sangat tinggi antara price, weekly_price, dan monthly_price (0.94 hingga 1.00) menunjukkan bahwa harga harian, mingguan, dan bulanan sangat berhubungan dan sejalan. Ketika harga harian meningkat, harga mingguan dan bulanan juga cenderung meningkat.
2. Korelasi Antar Variabel Skor Ulasan
Variabel skor ulasan seperti review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin review_scores_communication, review_scores_location, dan review_scores_value memiliki korelasi positif yang cukup tinggi satu sama lain.
Sebagai contoh:
- review_scores_rating memiliki korelasi sebesar 0.70 dengan review_scores_cleanliness, 0.65 dengan review_scores_checkin, dan 0.70 dengan review_scores_value.
Korelasi positif yang tinggi ini menunjukkan bahwa ketika suatu properti memiliki skor tinggi dalam satu aspek (misalnya cleanliness atau checkin), biasanya aspek lainnya juga mendapatkan skor tinggi. Ini dapat menunjukkan bahwa properti yang bagus dalam satu faktor cenderung berkinerja baik dalam faktor ulasan lainnya.
3. Hubungan Fasilitas Properti dengan Harga
accommodates, bathrooms, bedrooms, dan beds dengan price:
- Semua variabel ini memiliki korelasi positif moderat dengan price (0.53 hingga 0.63), menunjukkan bahwa semakin banyak kapasitas atau fasilitas yang dimiliki properti (lebih banyak kamar tidur, kamar mandi, atau tempat tidur), maka harga cenderung lebih tinggi.
guests_included dengan price:
- Korelasi positif antara guests_included dan price (0.46) menunjukkan bahwa harga cenderung lebih tinggi untuk properti yang mencakup lebih banyak tamu dalam tarif dasar.
4. Hubungan Skor Ulasan dengan Variabel Lain
review_scores_location dengan review_scores_value (0.37):
- Korelasi positif moderat menunjukkan bahwa properti yang memiliki nilai baik dalam location cenderung memiliki nilai tinggi juga dalam value.
review_scores_value dengan accommodates dan price (-0.07):
- Korelasi negatif kecil menunjukkan bahwa nilai ulasan value cenderung sedikit menurun jika kapasitas atau harga meningkat, yang bisa diartikan bahwa tamu merasa harga yang lebih tinggi atau kapasitas lebih besar tidak selalu sepadan.


**TOP 5 HARGA MEDIAN TERATAS**
![alt text](image-3.png)
Gambar ini menampilkan diagram batang yang menunjukkan lima Listing ID dengan harga median tertinggi dalam dataset. Judul grafik di bagian atas, "TOP 5 HARGA MEDIAN TERATAS," menunjukkan bahwa fokus grafik ini adalah pada lima listing dengan nilai median harga tertinggi.
Pada sumbu X, terdapat Listing ID yang merupakan identifikasi unik untuk setiap properti, yaitu 6119821, 6239108, 6387576, 6400379, dan 6403104. Sumbu Y menunjukkan harga median dari setiap listing dalam bentuk nilai numerik, dengan skala yang berkisar dari 0 hingga sekitar 100. Dari grafik ini, terlihat bahwa listing dengan ID 6387576 memiliki harga median tertinggi, diikuti oleh listing ID 6400379, sementara listing dengan ID 6119821 memiliki harga median terendah di antara lima listing yang ditampilkan.
Di bagian atas grafik, terlihat potongan kode Python dalam Jupyter Notebook yang menunjukkan cara pembuatan grafik ini menggunakan matplotlib.pyplot. Kode tersebut mengatur sumbu X dengan label "Listing ID" dan sumbu Y dengan label "Median Price," serta menambahkan judul untuk grafik. 


**TOP 5 LISTING BERDASARKAN HARGA TERTINGGI**
![alt text](image-4.png)
Gambar ini menampilkan diagram batang yang menunjukkan 5 Listing dengan Harga Tertinggi dalam dataset. Judul grafik, "Top 5 Listing Berdasarkan Harga Tertinggi," menjelaskan bahwa grafik ini menyajikan lima listing dengan nilai harga tertinggi.
Pada sumbu X, terdapat Listing ID untuk setiap properti, yaitu 2720963, 3308979, 4464824, 4825073, dan 5534463. Setiap batang mewakili harga dari masing-masing listing, yang diukur pada sumbu Y dengan label "Price". Sumbu Y menampilkan rentang harga dari 0 hingga lebih dari 1000.
Dari grafik ini, terlihat bahwa:
- Listing ID 3308979 dan 4825073 memiliki harga yang hampir sama dan termasuk dalam listing dengan harga tertinggi.
- Listing ID 5534463 memiliki harga yang lebih rendah dibandingkan empat listing lainnya tetapi masih termasuk dalam lima besar dengan harga tertinggi.