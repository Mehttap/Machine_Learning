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




