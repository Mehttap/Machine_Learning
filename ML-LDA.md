# Linear Discriminant Analysis - Doğrusal Diskriminant Analizi

LDA, gözetimli öğrenme kapsamında kullanılan istatistiklsel bir yöntem olup,özellikle sınıflandırma problemlerinde ve boyut indirgeme uygulamalarında yaygın olarak kullanılmaktadır. 
LDA'nın temel amacı, farklı sınıflara ait veri noktalarını mümkün olduğunca iyi ayırabilecek doğrusal bir projeksiyon bulunmaktadır.
Bu yöntemde veri, sınıflar arası ayrımı maksimize ederek sınıf içi varyansı minimize edecek şekilde daha düşük boyutlu bir uzaya dönüştürülür. 
Böylece hem veri görselleştirilmesi kolaylaşır hem de sınıflandırma algoritmalarının performansı arttırılabilir. 
Bu uygulamada, LDA yönteminin temel prensipleri incelenecek ve Python kullanılarak bir veriseti üzerinde pratik olarak uygulanacaktır. 

Amaç: Sınıfları birbirinden mümkün olduğunca iyi ayıran doğrusal projeksiyonu bulmak .

# Modül Tanımları

## Linear Discriminant Analysis Uygulaması

### LDA Yönetimi
### LDA'yı veri üzerinde uygulama 
### Sınıfları nasıl ayırdığını görselleştireceğiz. 

```
import numpy as np
import pandas as pd
import mat.plotlib.pyplot as plt
import seaborn as sns 

from sklearn.prespocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

```
## Veri setini yükleme 

```
df = sns.load_dataset ("penguins")

df.head()

```

## Veri setindeki bazı özellikler:

bill_length_mm --> gaga uzunlugu
bill_depth_mm -->  gaga derinliği
flipper_lenght_mm ---> yüzgeç uzunluğu
body_mass_g--> vucüt kütlesi
species---> penguen türü (hedef değişken)


```
df.info()

```
# Eksik Veri Kontrolü
```
df.isnull().sum

```
# Eksik Verileri Temizleyelim

```
df = df.dropna()
```
# Sınıf dağılımını inceleyelim

```
df["island"].unique()

```
# Veriyi Görselleştirme

Bazı özellikler zaten sınıfları kısmen ayırıyor. Ama: LDA sınıfları maksimum ayıracak yeni eksenler oluşturur. 

```
sns.pairplot(df, hue="species")
plt.show()

```

# Özellikler ve Hedef Değişkenin Ayrılması

X= df.drop(columns=["species", "island","sex"])

y= df["species"]

print(X.head())
print(y.head())

# Eğitim ve Test Verisi Ayırma

Verinin tamamını kullanmayız. Aksi halde overfitting olur. 

```
X_train, X_test, y_train, y_test =train_test_split(

    X,
y,
test_size=0.6,
random_state=42,
stratify=y )

print("Eğitim veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)
```
# LDA Modelini Eğitme

```
lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)
```
# LDA Modeli ile Tahmin

```
y_pred =lda.predict(X_test)

print(y_pred[:10]

```
# Model Performansı

```
accuracy = accuracy_score(y_test, y_pred)

print("Doğruluk (Accuracy):", accuracy)
```
# Karmaşıklık Matrisi

```
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")

plt.show()

```
# LDA ile Boyut İndirgeme

Eğer veri setinde C tane sınıf varsa, LDA en fazla C-1 boyuta indirilebilir. Penguin veri setinde: 3 sınıf ---> maksimum 2 boyut

```
lda_transform = LineardDiscriminantAnalysis(n_componenets=2)

X_lda = lda_transform.fit_transform(X,y)

print("Orijinal boyut:", X.shape)
print("Yeni boyut:", X_lda.shape)

```
# LDA Projeksiyonunu Görselleştirme

```
lda_df["LD1"] = X_lda[:,0]
lda_df["LD2"] = X_lda[:,1]
lda_df["species"] = y.values

```

```
sns. scatterplot(
data= lda_df,
x="LD1",
y="LD2",
hue="species",
s=88
)

plt.title("LDA Projeksiyonu")
plt.show()

```

















