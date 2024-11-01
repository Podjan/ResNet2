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
1. **Listing ID**:
Rentang nilai Listing ID sangat lebar, mengindikasikan banyaknya listing yang ada. Tidak ada outlier yang signifikan, menunjukkan distribusi data yang cukup merata.
2. **Tahun**:Sebagian besar data terkonsentrasi pada tahun 2016 dan 2017.
Tidak ada outlier yang terlihat.
3. **Bulan**: Distribusi data cukup merata sepanjang tahun, meskipun mungkin ada sedikit fluktuasi musiman yang tidak terlalu terlihat jelas dari boxplot ini.
4. **Bulan_num**: Variabel ini kemungkinan merupakan representasi numerik dari bulan (1-12). Distribusi data sangat merata, seperti yang diharapkan.
5. **Median_price**: Terdapat variasi harga yang cukup besar, dengan median harga yang relatif rendah.
Adanya beberapa outlier di bagian atas menunjukkan adanya beberapa listing dengan harga yang jauh di atas rata-rata.
Secara umum, boxplot ini menunjukkan bahwa harga sewa Airbnb di Seattle bervariasi cukup signifikan.
6. **Jumlah_id**: Variabel ini mungkin mewakili jumlah sesuatu (misalnya, jumlah ulasan, jumlah kamar).
Distribusi data cenderung condong ke kiri (negatively skewed), dengan sebagian besar nilai terkonsentrasi di bagian bawah.
#### df2

### Univariate Analysis
#### df1
1. **Listing ID**:
Rentang nilai Listing ID sangat lebar. Tidak ada outlier yang signifikan.

3. **Tahun**: Data sebagian besar ada di 2016 akan tetapi ada outlier pada 2017 yang harus di alaissi lebih lanjut
   
5. **Bulan**: Distribusi data cukup merata sepanjang tahun, yaitu dari bulan maret hingga Juni
   
6. **Bulan_num**: Distribusi data sangat merata.
   
7. **Median_price**: Terdapat variasi harga yang cukup besar, dengan median harga yang relatif rendah. Adanya beberapa outlier di bagian atas menunjukkan adanya beberapa listing dengan harga yang jauh di atas rata-rata.
   
8. **Jumlah_id**: Distribusi data cenderung condong ke kiri (negatively skewed), dengan sebagian besar nilai terkonsentrasi di bagian bawah.
#### df2
##### nums21
1. **Jumlah_id**: Distribusi terpusat di sekitar angka 300, menunjukkan jumlah "id" yang konsisten antar data. Kemungkinan mewakili jumlah ulasan atau interaksi, dengan nilai tinggi mengindikasikan listing yang populer.

2. **host_response_rate**: Banyak host dengan tingkat respons tinggi, tetapi ada juga yang sangat rendah. Hal ini menunjukkan bahwa responsivitas host cukup beragam, dengan yang cepat lebih disukai tamu.

3. **host_acceptance_rate**: Mayoritas host memiliki tingkat penerimaan mendekati 100%, tetapi ada juga yang lebih selektif. Host dengan penerimaan tinggi cenderung populer atau memiliki kebijakan khusus.

4. **accommodates**: Sebagian besar listing menampung hingga 4 orang, dengan beberapa outlier yang menampung lebih banyak. Kapasitas akomodasi beragam, dari 1-2 hingga lebih dari 8 orang.

5. **bathrooms**: Sebagian besar listing memiliki 1-2 kamar mandi, dengan beberapa outlier yang memiliki lebih banyak. Fasilitas kamar mandi bervariasi, meski mayoritas terbatas.

##### nums22
1. **Bedrooms**: Sebagian besar listing memiliki 1-2 kamar tidur, cocok untuk individu atau keluarga kecil. Listing dengan lebih banyak kamar tidur biasanya untuk kelompok besar atau masa inap lebih lama.

2. **Beds**: Umumnya ada 2-4 tempat tidur per listing, dengan beberapa memiliki lebih banyak. Beberapa listing menawarkan tempat tidur tambahan, seperti sofa bed.

3. **Price**: Harga per malam bervariasi dengan median rendah, namun ada outlier dengan harga tinggi, dipengaruhi oleh lokasi, ukuran, dan fasilitas.

4. **Weekly_price**: Harga mingguan mirip dengan harga harian, biasanya kelipatan dengan diskon atau biaya tambahan.

5. **Monthly_price**: Harga bulanan cenderung lebih murah per hari dibandingkan harga harian, sering kali disertai diskon.

##### nums23
1. **guests_included**: Sebagian besar listing dapat menampung hingga 4 tamu, dengan beberapa yang bisa menampung lebih dari 8 orang. Listing dengan kapasitas besar cocok untuk keluarga atau kelompok besar.

2. **minimum_nights**: Minimum menginap umumnya rendah (1-2 malam), dengan beberapa listing yang mensyaratkan durasi lebih lama untuk masa inap jangka panjang.

3. **maximum_nights**: Mayoritas listing memiliki batas maksimum menginap yang fleksibel atau tanpa batas, namun ada beberapa yang membatasi masa inap lebih pendek.

4. **review_scores_rating**: Sebagian besar listing memiliki rating tinggi (di atas 80), menunjukkan kualitas yang baik dan tamu yang puas.

5. **review_scores_accuracy**: Skor akurasi deskripsi umumnya tinggi (di atas 8), menunjukkan deskripsi listing yang sesuai dengan ekspektasi tamu.

##### nums24
1. **review_scores_cleanliness**: Skor kebersihan mayoritas tinggi (di atas 8), menunjukkan host menjaga standar kebersihan yang baik.

2. **review_scores_checkin**: Skor check-in umumnya tinggi, meski ada beberapa kesulitan di beberapa listing.

3. **review_scores_communication**: Skor komunikasi juga tinggi, menandakan host responsif dan mudah diajak berkomunikasi.

4. **review_scores_location**: Skor lokasi tinggi, menunjukkan lokasi listing umumnya strategis dan mudah diakses.

5. **review_scores_value**: Skor nilai tinggi, menunjukkan tamu merasa puas dengan harga yang dibayar sesuai fasilitas yang diterima.

### Multivariate Analysis
#### df1

#### df2
