import pandas as pd  # Pandas kutuphanesi veri analizi icin kullanilir
import numpy as np   # NumPy kutuphanesi sayisal hesaplamalar icin kullanilir
import tensorflow as ts   # TensorFlow makine ogrenimi ve derin ogrenme kutuphanesidir
import matplotlib.pyplot as plt   # Veri gorsellestirme icin kullanilir
import seaborn as sbn   # Matplotlib tabanli daha cezbedici gorsellestirmeler icin kullanilir

# Veri setini Excel dosyasindan oku
data = pd.read_excel("merc.xlsx")

# Veri setinin baslangic degerlerini goruntule
data.head()

# Veri setinin temel istatistik bilgilerini goruntule
data.describe()

# Veri setindeki eksik degerlerin sayisini goruntule
data.isnull().sum()

# "price" degiskeninin dagilimini gorsellestir
sbn.distplot(data["price"])

# "year" degiskenine gore arac sayisini gorsellestir
sbn.countplot(x="year", data=data)

# Sadece sayisal verileri sec
numeric_data = data.select_dtypes(include=["float64", "int64"])

# Sayisal veriler arasindaki korelasyon matrisini hesapla
numeric_data.corr()

# "price" ile diger sayisal degiskenler arasindaki korelasyonlari goster
numeric_data.corr()["price"].sort_values()

# "mileage" ve "price" arasindaki iliskiyi gorsellestir
sbn.scatterplot(x="mileage", y="price", data=data)

# En yuksek fiyatli 20 araci goruntule
data.sort_values("price", ascending=False).head(20)

# Veri setinin %1'ini temizle (outlierlari kaldir)
cleandata = data.sort_values("price", ascending=False).iloc[131:]

# Temizlenmis veri setinin temel istatistik bilgilerini goruntule
cleandata.describe()

# "year" degiskenine gore gruplayarak ortalama fiyatlari goruntule
data.groupby("year")["price"].mean()

# Outlier'lari temizledikten sonra "year" degiskenine gore gruplayarak ortalama fiyatlari goruntule
cleandata.groupby("year")["price"].mean()

# 1970 model aracları veri setinden cikar
data = cleandata
data = data[data.year != 1970]

# "transmission" sutununu veri setinden cikar
data = data.drop("transmission", axis=1)

# Bagimli degiskeni (price) ve bagimsiz degiskenleri (diger ozellikler) ayir
y = data["price"].values
x = data.drop("price", axis=1).values

# Veriyi egitim ve test setlerine ayir
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Ozellikleri olceklendir
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# TensorFlow ve Keras ile bir model olustur
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1))

# Modeli derle
model.compile(optimizer="adam", loss="mse")

# Modeli egit
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=300)

# Egitim surecindeki kayiplari (loss) goruntule
lostData = pd.DataFrame(model.history.history)
lostData.plot()

# Modelin performansini degerlendir
from sklearn.metrics import mean_absolute_error, mean_squared_error
guessArray = model.predict(x_test)
mean_absolute_error(y_test, guessArray)
