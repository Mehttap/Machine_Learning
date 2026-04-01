# Principal Component Analysis - Temel Bileşen Analizi 

PCA, yüksek boyutlu verileri daha düşük boyutlu bir uzaya indirgemek için kullanılan bir boyut indirgeme yöntemidir.
PCA'nın temel amacı: verideki varyansın mümkün olan en büyük kısmını koruyarak yeni eksenler (principal components) oluşturmaktır.
Bu eksenler birbirine diktir (orthogonal) ve verinin en fazla varyansa sahip olduğu yönleri temsil eder. PCA özellikle veri
görselleştirme, gürültü azaltma ve makine öğrenmesi modellerini hızlandırmak amacıyla yaygın olarak kullanılmaktadır.

## Gerekli Kütüphaneler

``` import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
## Veri Setini Yükleme

### İris veri setini yükle

``` iris = load_iris()

X = iris.data
Y = iris.target

feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

df.head(10)
```

```
df.shape
```

### Kaç farklı hedef değişken var
```
df["target"].unique()
```



# Veri Görselleştirme (PCA Öncesi)

## Pairplot ile veri ilişkilerine bakmak


```
sns.pairplot(df, hue="target")
plt.show()
```

# Veriyi Standardize Etme

Neden stadardize ediyoruz? Çünkü PCA variance based bir yöntemdir. 

## PCA öncesi veriyi ölçeklendirmeliyiz

```
scaler = StandardScaler()

X_scaled = scaler.fit_transform(x)

print("Ortalama:" , np.mean(X_scaled, axis=0))
print("Standart sapma:" , np.std(X_scaled, axis=0))
```
## PCA modelini oluştur

```
pca = PCA()

X_pca = pca.fit_transform (X_scaled)

print("Yeni veri boyutu:" , X_pca.shape)
```
hala 4 bileşen var. PCA sadece eksenleri değiştirdi.

# Açıklanan Varyans (En Önemli PCA Grafiği)

## Açıklanan varyans oranları

```
explained_variance = pca.explained_variance_ratio_

print("Explained variance ratio:")
print(explained_variance)

```

PC1
PC2
PC3
PC4
Genellikle ilk iki bileşen yeterli olur. 

```
plt.figure(figsize=(6,4))

plt.plot(range(1,5), explained_variance, marker='o')
plt.xlabel("Temek Bileşen")
plt.ylabel("Açıklanan Varyans Oranı")
plt.title("Screen Grafiği")

plt.show()

```

# PCA İLE 2 BOYUTA DÜŞÜRME

4boyut--> 2 boyut

## 2 Bileşenli PCA

```
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(X_pca[:5])

```
# PCA Sonrası Görselleştirme

## PCA sonuçlarını dataframe yapalım

```
pca_df = pd.DataFrame()

pca_df["PC1"] = X_pca[:,0]
pca_df["PC2"] = X_pca[:,"]
pca_df["target"] = y

```

```
plt.figure(figsize=(6,5))

sns.scatterplot(
x= "PC1",
y= "PC2",
hue ="target",
data=pca_df,
palette="Set1"
)

plt.title("PCA ile 2 Boyuta İndirgenmiş Veri")
plt.show()

```
# PCA Bileşenleri

PCi =  w1x1+ w2x2+...+wnxn

```
componenets = pca.componenets_

print("PCA Bileşenleri:")
print(componenets)

```
Yani, 4 adet olan özellik vektörünü sadece 2 adet olacak şekilde ve en az veri kaybı ile sıkıştırmış olduk.
Depolanması ve üzerine işlem yapılması gereken veri büyüklüğü yarıya inerken, %94 veri koruması sağlandı (screen grafiği ilk 2 komponenet toplamı).













