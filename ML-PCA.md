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









