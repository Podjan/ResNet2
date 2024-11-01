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
Pada proses descriptive statistic ini, berguna untuk melihat nilai statistika dari setiap data yang sudah diolah sebelumnya. Pada bagian ini descriptive statistic menampilkan 2 data frame yang berguna untuk mempertimbangkan analisis selanjutnya.
#### df1
df1 merupakan data frame yang data sourcenya diambil dari dataset calendar. Sebelum dilakukan data cleansing terdapat sekitar 1.393.570 rows yang kemudian diolah dengan membuang missing valuenya sehingga feature payment yang awalnya 459028 didrop semua menjadi 0.

 | Features                 | Before | After |
 |--------------------------|--------|-------|
 | payment                  | 459028 | 0     |

 
Salah satu pertimbangan melakukan drop terhadap missing value adalah karena listing yang available saat itu berstatus false artinya pada saat itu penginapannya tidak available dan tidak ada proses transaksi pada listing tersebut. Hal ini sangat mempengaruhi proses selanjutnya mengingat goal yang ingin dicapai yaitu memprediksi jumlah pengunjung di tahun berikutnya dan memprediksi harga hotel di tahun depan, jadi membersihkan data null sangatlah penting dalam proses ini.    

Seperti yang diterangkan sebelumnya, data selanjutnya di groupby untuk mendapatkan nilai median price dan jumlah customer. Sehingga total keseluruhan data df1 setelah di drop adalah 36.115 rows. Pada akhirnya standar analisis yang digunakan mengacu pada data booking perbulan di masing-masing listing atau penginapan.
 

Adapun hasil descriptive statistic nya adalah sbb:
| Statistic    | listing_id       | tahun   | bulan_num | median_price | jumlah_id |
|--------------|------------------|---------|-----------|--------------|-----------|
| count        | 36115.000000     | 36115.0 | 36115.0   | 36115.000000 | 36115.000 |
| mean         | 5350481.000000   | 2016.08 | 5.98      | 135.79       | 25.88     |
| std          | 2974714.000000   | 0.27    | 3.66      | 103.64       | 9.36      |
| min          | 333500.000000    | 2016.0  | 1.0       | 20.0         | 1.0       |
| 25%          | 2933877.000000   | 2016.0  | 3.0       | 75.0         | 28.0      |
| 50%          | 5691933.000000   | 2016.0  | 6.0       | 105.0        | 30.0      |
| 75%          | 7915432.000000   | 2016.0  | 9.0       | 158.0        | 31.0      |
| max          | 10340160.000000  | 2017.0  | 12.0      | 1650.0       | 31.0      |

Dari hasil descriptive statistic tidak menunjukkan adanya data yang invalid dan sebagian besar datanya berasal dari tahun 2016 dengan rata-rata distribusi datanya berada dipertengahan bulan sekitar Mei-Juni. median_price yang tinggi dan standar deviasi besar mengindikasikan adanya listing dengan harga sangat tinggi yang mempengaruhi rata-rata harga.


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
df2 adalah data frame yang data source nya diambil dari dataset listing_id. data ini sangat berguna dalam analisis tingkat kepuasan customer. 

Pada df2 total data sebelum diolah ada sekitar 3818 rows listing penginapan. Selanjutnya, data terebut diolah dengan melakukan groupby berdasarkan listing_id untuk mendapatkan jumlah_id dari masing-masing listing.
```python
dfcg3 = dfcg2.groupby(['listing_id']).agg(jumlah_id=('jumlah_id', 'sum')).reset_index()
```
Sehingga dari hasil tersebut didapatkan total keseluruhan row adalah 3723 dengan 37 kolom. Adapun kolom-kolom yang masih memiliki missing value adalah sebagai berikut:

   | Features                     | tot. data nan    |
   |------------------------------|------------------|
   | host_response_time           | 480              |
   | host_response_rate           | 480              |
   | host_acceptance_rate         | 723              |
   | host_is_superhost            | 2                |
   | host_identity_verified       | 2                |
   | zipcode                      | 7                |
   | property_type                | 1                |
   | bathrooms                    | 16               |
   | bedrooms                     | 6                |
   | beds                         | 1                |
   | weekly_price                 | 1750             |
   | monthly_price                | 2231             |
   | review_scores_rating         | 623              |
   | review_scores_accuracy       | 633              |
   | review_scores_cleanliness    | 628              |
   | review_scores_checkin        | 633              |
   | review_scores_communication  | 626              |
   | review_scores_location       | 630              |
   | review_scores_value          | 631              |
   
Adapun missing value tersebut akan diolah lebih lanjut ditahap berikutnya. 

Berikut hasil descriptive statistic dari df2:
| Statistic                    | count     | mean          | std          | min      | 25%      | 50%      | 75%      | max       |
|------------------------------|-----------|---------------|--------------|----------|----------|----------|----------|-----------|
| listing_id                   | 37230.0   | 5.548051e+06 | 2.969790e+06 | 3335.0   | 3242426.0| 6119821.0| 8036802.5| 10340165.0|
| jumlah_id                    | 37230.0   | 2.510518e+02 | 1.221917e+02 | 1.0      | 130.0    | 310.0    | 360.0    | 365.0     |
| host_response_rate           | 32430.0   | 9.484548e+01 | 1.192436e+01 | 17.0     | 90.0     | 100.0    | 100.0    | 100.0     |
| host_acceptance_rate         | 30000.0   | 9.966667e+01 | 1.825742e+00 | 0.0      | 100.0    | 100.0    | 100.0    | 100.0     |
| accommodates                 | 37230.0   | 3.330065e+00 | 2.035758e+00 | 1.0      | 2.0      | 3.0      | 4.0      | 16.0      |
| bathrooms                    | 37070.0   | 1.258611e+00 | 8.578998e-01 | 0.0      | 1.0      | 1.0      | 2.0      | 8.0       |
| bedrooms                     | 37170.0   | 1.297812e+00 | 8.749494e-01 | 0.0      | 1.0      | 1.0      | 2.0      | 7.0       |
| beds                         | 37220.0   | 1.728641e+00 | 1.133858e+00 | 0.0      | 1.0      | 2.0      | 2.0      | 20.0      |
| price                        | 37230.0   | 1.359909e+02 | 1.036437e+02 | 10.0     | 70.0     | 105.0    | 158.0    | 1650.0    |
| weekly_price                 | 19730.0   | 7.877248e+02 | 5.340506e+02 | 0.0      | 455.0    | 600.0    | 910.0    | 6300.0    |
| monthly_price                | 14920.0   | 2.604846e+03 | 1.724425e+03 | 0.0      | 1300.0   | 2000.0   | 3000.0   | 12000.0   |
| guests_included              | 37230.0   | 1.590279e+00 | 1.119754e+00 | 1.0      | 1.0      | 1.0      | 2.0      | 16.0      |
| minimum_nights               | 37230.0   | 2.365556e+01 | 6.513065e+01 | 1.0      | 2.0      | 3.0      | 5.0      | 1250.0    |
| maximum_nights               | 37230.0   | 7.859063e+02 | 1.702415e+03 | 1.0      | 60.0     | 1125.0   | 1125.0   | 100000.0  |
| review_scores_rating         | 31010.0   | 9.453548e+01 | 6.600270e+01 | 20.0     | 93.0     | 96.0     | 100.0    | 100.0     |
| review_scores_accuracy       | 30950.0   | 9.505821e+00 | 7.494370e-01 | 2.0      | 9.0      | 10.0     | 10.0     | 10.0      |
| review_scores_cleanliness    | 30950.0   | 9.484342e+00 | 8.105796e-01 | 3.0      | 9.0      | 10.0     | 10.0     | 10.0      |
| review_scores_checkin        | 30990.0   | 9.788350e+00 | 5.944549e-02 | 9.0      | 10.0     | 10.0     | 10.0     | 10.0      |
| review_scores_communication  | 31010.0   | 9.762451e+00 | 5.733428e-01 | 4.0      | 10.0     | 10.0     | 10.0     | 10.0      |
| review_scores_location       | 30990.0   | 9.607348e+00 | 7.547469e-01 | 2.0      | 9.0      | 10.0     | 10.0     | 10.0      |
| review_scores_value          | 30920.0   | 9.543428e+00 | 7.494385e-01 | 2.0      | 9.0      | 10.0     | 10.0     | 10.0      |

Dari data tersebut tidak menunjukkan adanya data yang invalid. Selain itu, beberapa point yang bisa kita dapat dari data tesebut adalah :

1. Nilai rata-rata jumlah_id berkisar 251 dengan variasi sebesar 122 dan median 310, artinya beberapa listing memiliki sebagian besar jumlah_id nya 310 yang mana itu mengindikasikan bahwa jumlah_id memilik median > mean sehingga distribusinya cenderung negative skewed
2. Untuk nilai kepuasan didapatkan dari  kolom host_response_rate, host_acceptance_rate, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, menunjukkan skor rata-rata untuk semua aspek mendekati nilai maksimum (9,5-10) yang artinya kebanyakan listing memiliki ulasan yang sangat positif.
3. Pada kolom price menunjukkan adanya perbedaan harga minimum dan maximumnya yang cukup jauh, artinya ada listing dengan harga yang sangat mahal dan sangat murah. Ini menandakan adanya outlier pada data ini. Selain itu, sebagian besar listing price per malamnya berkisar $105. 
4. Accomodates menunjukkan jumlah tamu yang dapat ditampung oleh suatu listing. Rata-rata kapasitas yang bisa ditampung adalah sekitar 3 tamu dengan sebagian besar listing mampu menampung 3-4 tamu.
5. Sebagian besar listing hanya memiliki satu kamar mandi, satu kamar tidur, dan satu hingga dua tempat tidur. Namun beberapa listing memiliki jumlah fasilitas yang tinggi, dengan nilai maksimum 8 kamar mandi, 7 kamar tidur, dan 20 tempat tidur.



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