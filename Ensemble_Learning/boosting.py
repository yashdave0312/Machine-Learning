import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

df = sns.load_dataset("iris")
print(df.head())

X = df.drop("species",axis =1)
y = df["species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42)

ada_model = AdaBoostClassifier(n_estimators=100,random_state=42)

ada_model.fit(X_train,y_train)

y_pred = ada_model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)