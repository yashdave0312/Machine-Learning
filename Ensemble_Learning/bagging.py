import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
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

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42,stratify =y)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42, 
)

rf_model.fit(X_train,y_train)

print(rf_model)

y_pred = rf_model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# cv = cross_val_score(rf_model,X,y_encoded,cv = 5,scoring = "accuracy")
# print(cv)