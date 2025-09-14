
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

def main():
    os.makedirs("project3_ml_math", exist_ok=True)
    base_dir = "project3_ml_math"
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.data; y = cancer.target
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42,stratify=y)

    logreg = LogisticRegression(max_iter=5000,random_state=42).fit(X_train,y_train)
    y_pred_log = logreg.predict(X_test)
    y_prob_log = logreg.predict_proba(X_test)[:,1]
    acc_log = accuracy_score(y_test,y_pred_log)
    fpr_log,tpr_log,_=roc_curve(y_test,y_prob_log); roc_auc_log=auc(fpr_log,tpr_log)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_scores_log=cross_val_score(logreg,X_scaled,y,scoring="accuracy",cv=cv)

    dt=DecisionTreeClassifier(max_depth=4,random_state=42).fit(X_train,y_train)
    y_pred_dt=dt.predict(X_test); y_prob_dt=dt.predict_proba(X_test)[:,1]
    acc_dt=accuracy_score(y_test,y_pred_dt)
    fpr_dt,tpr_dt,_=roc_curve(y_test,y_prob_dt); roc_auc_dt=auc(fpr_dt,tpr_dt)
    cv_scores_dt=cross_val_score(dt,X_scaled,y,scoring="accuracy",cv=cv)

    metrics=pd.DataFrame([
        {"Model":"Logistic Regression","Accuracy":acc_log,"AUC":roc_auc_log,"CV_Accuracy_Mean":cv_scores_log.mean(),"CV_Accuracy_SD":cv_scores_log.std()},
        {"Model":"Decision Tree","Accuracy":acc_dt,"AUC":roc_auc_dt,"CV_Accuracy_Mean":cv_scores_dt.mean(),"CV_Accuracy_SD":cv_scores_dt.std()}
    ])
    metrics.to_csv(os.path.join(base_dir,"metrics_summary.csv"),index=False)

    plt.figure()
    plt.plot(fpr_log,tpr_log,label=f"Logistic (AUC={roc_auc_log:.2f})")
    plt.plot(fpr_dt,tpr_dt,label=f"Decision Tree (AUC={roc_auc_dt:.2f})")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Logistic Regression vs Decision Tree"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir,"fig_roc_curves.png"),dpi=200); plt.close()

    pca=PCA(n_components=2); X_pca=pca.fit_transform(X_scaled)
    X_train_pca,X_test_pca,y_train_pca,y_test_pca=train_test_split(X_pca,y,test_size=0.2,random_state=42,stratify=y)
    logreg2d=LogisticRegression(max_iter=5000,random_state=42).fit(X_train_pca,y_train_pca)
    xx,yy=np.meshgrid(np.linspace(X_pca[:,0].min()-1,X_pca[:,0].max()+1,200),np.linspace(X_pca[:,1].min()-1,X_pca[:,1].max()+1,200))
    Z=logreg2d.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    plt.figure(); plt.contourf(xx,yy,Z,alpha=0.3)
    plt.scatter(X_pca[:,0],X_pca[:,1],c=y,edgecolor='k',s=20)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Decision Boundary (Logistic, PCA)")
    plt.tight_layout(); plt.savefig(os.path.join(base_dir,"fig_decision_boundary.png"),dpi=200); plt.close()

    importances=pd.DataFrame({"feature":cancer.feature_names,"importance":dt.feature_importances_}).sort_values("importance",ascending=False)
    importances.to_csv(os.path.join(base_dir,"feature_importances.csv"),index=False)
    plt.figure(); plt.bar(importances["feature"].head(10),importances["importance"].head(10))
    plt.xticks(rotation=90); plt.title("Top 10 Feature Importances (Decision Tree)")
    plt.tight_layout(); plt.savefig(os.path.join(base_dir,"fig_feature_importances.png"),dpi=200); plt.close()

if __name__=="__main__":
    main()
