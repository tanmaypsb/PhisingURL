from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from .data_cleaning import TARGET

def train_models(data):
    X=data.drop(columns=TARGET); y=data[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={
        "DummyClassifier":DummyClassifier(strategy="most_frequent"),
        "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42))]),
        "Decision Tree":DecisionTreeClassifier(class_weight="balanced",max_depth=12,min_samples_leaf=5,random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=250,class_weight="balanced_subsample",min_samples_leaf=2,n_jobs=-1,random_state=42),
    }
    fitted={}
    for name,model in models.items():
        model.fit(X_train,y_train); fitted[name]=model
    return X_train,X_test,y_train,y_test,fitted
