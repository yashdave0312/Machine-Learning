import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
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

base_learners = [
    ("dt",DecisionTreeClassifier(random_state=42)),
    ("lr",LogisticRegression(max_iter=1000)),
    ("svc",SVC(probability=True,kernel="rbf",random_state=42))
]

# base_learners = [
#     ("dt",DecisionTreeClassifier()),
#     ("lr",LogisticRegression()),
#     ("svc",SVC())
# ]

print(base_learners)

meta_learners = LogisticRegression()

stacking_el = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learners,
    cv=5
)

stacking_el.fit(X_train,y_train)

print(stacking_el)

y_pred = stacking_el.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)