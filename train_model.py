import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Facebook Metrics of Cosmetic Brand.csv')

print("Colonnes du DataFrame :", df.columns)

df['Type'] = df['Type'].astype('category').cat.codes
df['Category'] = df['Category'].astype('category').cat.codes

# Caractéristiques et cibles
X = df[['Page total likes', 'Type', 'Category', 'Post Month',
        'Post Weekday', 'Post Hour', 'Paid', 'Lifetime Post Total Reach',
        'Lifetime Post Total Impressions', 'Lifetime Engaged Users',
        'Lifetime Post Consumers', 'Lifetime Post Consumptions',
        'Lifetime Post Impressions by people who have liked your Page',
        'Lifetime Post reach by people who like your Page',
        'Lifetime People who have liked your Page and engaged with your post']]
         
# Cibles : prédire le nombre de likes
y = df['Total Interactions']  

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, 'model_popularite_post.pkl')



