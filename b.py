import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

# 1. Load the data
f = pd.read_csv("melb_data.csv")

# 2. Review the data
print(f.head(10))
print(f.columns)
print(f.shape)
print(f.describe())
f.hist(figsize=(10,6),bins=50,color="purple")
plt.tight_layout()
plt.show()
print(f.describe())

# 3. checking the data 
print(f.isna().sum())
print((f.isnull().sum()).sort_values(ascending=False))
print(f.duplicated().sum())
# After reviewing the data we have found missing values in columns [car,biulding_area,yearbiult,council area].

# 4. preparing the data
print(f.dtypes)
num_col=["Car","BuildingArea","YearBuilt"]
tex_col=["CouncilArea"]
num_imputer=SimpleImputer(strategy="median")
text_imputer = SimpleImputer(strategy="most_frequent")
f[num_col]= num_imputer.fit_transform(f[num_col])
f[tex_col] = text_imputer.fit_transform(f[tex_col]) 
print(f.isna().sum())

# 5. Feature engeneering
# 5.1 building ratio
f["build_ratio"] = f["Landsize"] / (f["BuildingArea"] + 1)
# 5.2 Biullding location
f["Distance_zone"]=pd.cut(f["Distance"],
                          bins=[5,10,15,20,50],
                          labels=["Very_close","Close","Medium","Far"])
f["is_close_cbd"] = (f["Distance"]<=10).astype(int)
# the Era of the building
f["era"] = pd.cut(f["YearBuilt"],
                  bins=[1800, 1920, 1960, 1990, 2010, 2025],
                  labels=["heritage", "mid_century", "modern", "contemporary", "new"])
f = f.replace([np.inf, -np.inf], np.nan)
f = f[f["Landsize"] < 10000]
f = f[f["BuildingArea"] < 1000]

# the date
f["Date"] = pd.to_datetime(f["Date"], dayfirst=True)
f["Year"] = f["Date"].dt.year
f["Month"] = f["Date"].dt.month
f["Age"] = 2025 - f["YearBuilt"]
f = f.drop("Date", axis=1)

# 6. Analyze the data
plt.figure(figsize=(15,10))
sns.heatmap(f.select_dtypes(include="number").corr(),
            cmap="magma",
            annot=True,
            fmt=".2f",)
plt.title("Correlation Heatmap")
plt.show()

fig,ax = plt.subplots(3,3,figsize=(20,13))
ax=ax.flatten()

# 6.1 Price distribution
sns.histplot(f["Price"], bins=50, kde=True, ax=ax[0], color="purple")
ax[0].set_title("Price Distribution")

# 6.2. Price by Region
region_price = f.groupby("Regionname")["Price"].median().sort_values()
sns.barplot(x=region_price.values, y=region_price.index, ax=ax[1], palette="magma")
ax[1].set_title("Median Price by Region")

# 6.3 building area
sns.scatterplot(x="BuildingArea", y="Price", data=f, ax=ax[2], alpha=0.3, color="purple")
ax[2].set_title("Building Area vs Price")

# 6.4 Age vs Price
sns.scatterplot(x="Age", y="Price", data=f, ax=ax[3], alpha=0.3, color="purple")
ax[3].set_title("Property Age vs Price")

# 6.5 Rooms Distribution
sns.countplot(x="Rooms", data=f, ax=ax[4], palette="magma")
ax[4].set_title("Rooms Distribution")

# 6.6 Property type distribution
sns.countplot(x="Type", data=f, ax=ax[5], palette="magma")
ax[5].set_title("Property Type Distribution")

# 6.7 Distance vs Price
sns.scatterplot(x="Distance", y="Price", data=f, ax=ax[6], alpha=0.3, color="purple")
ax[6].set_title("Distance vs Price")

# 6.8 Price by Era
sns.boxplot(x="era", y="Price", data=f, ax=ax[7], palette="magma")
ax[7].set_title("Price by Era")

# 6.9 Pice Trend over year
year_price = f.groupby("Year")["Price"].median()
sns.lineplot(x=year_price.index, y=year_price.values, ax=ax[8])
ax[8].set_title("Median Price by Year")
plt.tight_layout()
plt.show()

# 7. splitting the data 
f = f.drop(["Address", "Postcode"], axis=1)
x = f.drop(["Price"],axis = 1)
y = np.log1p(f["Price"])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
categorical = ["Suburb", "Type", "Method", "SellerG",
               "CouncilArea", "Regionname", "Distance_zone", "era"]
one_hot= OneHotEncoder(handle_unknown="ignore")
transformer = ColumnTransformer([("one_hot",
                                one_hot,
                                categorical)],
                                remainder="passthrough")
x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

# 8. The model
models = {
  "XGBRegressor":XGBRegressor(),
  "LGBMRegressor":LGBMRegressor(),
  "CatBoostRegressor":CatBoostRegressor(),
  "RandomForestRegressor":RandomForestRegressor(),
  "GradientBoostingRegressor":GradientBoostingRegressor(),
  "LinearRegression":LinearRegression(),
  "Ridge":Ridge(),
  "Lasso":Lasso(),
  "KNeighborsRegressor":KNeighborsRegressor()}

# 9. training and testing the model
result = []
best_pipe=None
best_name= None
R2= -999
Meansquarederror = 0
Rootsquarederror = 0
for model,m in models.items():
  pipe = Pipeline([
    ("scale",StandardScaler(with_mean=False)),
    ("model",m),])

  pipe.fit(x_train,y_train)
  y_pred = pipe.predict(x_test)
  
  r2 = r2_score(y_test,y_pred)
  meansquarederror = mean_squared_error(y_test,y_pred)
  root = np.sqrt(meansquarederror)
  
  print(f"-------{model}-------")
  print("R2Score",r2)
  print("Mean_Squared_error",meansquarederror)
  print("Root_Mean_Squared_Error",root)
  
  result.append({
    "Name":model,
    "pipe":pipe,
    "R2":r2,
    "Mean_squared_error":meansquarederror,
    "RootSquarederror":root})
  
  if r2 > R2 :
    best_name = model
    best_pipe = pipe
    R2 = r2
    Meansquarederror = meansquarederror
    Rootsquarederror =root
    
# 10. the Best Model
print("---------------")   
print("The best Model is:",best_name)
print("R2:", R2)
print("Mean_Squared_Error:", Meansquarederror)
print("Root_Mean_Squared_Error:", Rootsquarederror)

# 11. tune the model 
grid_= {
    "model__iterations": [500, 1000],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__depth": [4, 6, 8],
    "model__l2_leaf_reg": [1, 3, 5]}

tune_ = RandomizedSearchCV(best_pipe,
                           grid_,
                           n_iter = 15,
                           cv = 5,
                           n_jobs=-1,
                           scoring = "r2",
                           random_state= 42)
tune_.fit(x_train,y_train)
print("Best params:", tune_.best_params_)
print("Best CV R2:", tune_.best_score_)

y_pred_tuned = tune_.best_estimator_.predict(x_test)
print("Test R2:", r2_score(y_test, y_pred_tuned))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_tuned)))

grid_2={
  "model__iterations":[800,1000,1200],
  "model__learning_rate":[0.08,0.1,0.12],
  "model__depth":[5,6,7],
  "model__l2_leaf_reg":[2,3,4]}

tune_2=GridSearchCV(best_pipe,
                    grid_2,
                    cv=5,
                    scoring="r2",
                    n_jobs=-1)
tune_2.fit(x_train,y_train)
print("Best Params:",tune_2.best_params_)
print("Best CV R2:",tune_2.best_score_)

y_pred_tuned2 = tune_2.best_estimator_.predict(x_test)
print("Test R2:",r2_score(y_test,y_pred_tuned2))
print("RSME:",np.sqrt(mean_squared_error(y_test,y_pred_tuned2)))

# 12. Analyze the model
y_pred_best = tune_2.best_estimator_.predict(x_test)
y_pred_actual = np.expm1(y_pred_best)
y_test_actual = np.expm1(y_test)

fig,ax = plt.subplots(1,3,figsize=(18,5))
# 12.1 Actual Vs Predicted
ax[0].scatter(y_test_actual,y_pred_actual,alpha=0.3,color="purple")
ax[0].plot([y_test_actual.min(),y_test_actual.max()],
           [y_test_actual.min(),y_test_actual.max()],
           color="red",linewidth=2)
ax[0].set_xlabel("Actual Price")
ax[0].set_ylabel("Predicted Price")
ax[0].set_title("Actual Vs Predicted")

# 12.2 Residuals Distribution
residuals = y_test_actual - y_pred_actual
ax[1].hist(residuals,bins=50,color="purple")
ax[1].axvline(x=0, color="red", linewidth=2)
ax[1].set_title("Residuals Distribution")
ax[1].set_xlabel("Residual")

feature_names = transformer.get_feature_names_out()
importances = tune_2.best_estimator_.named_steps["model"].feature_importances_
feat_df = pd.DataFrame({"Feature":feature_names,"Importance":importances})
feat_df = feat_df.sort_values("Importance",ascending=False).head(15)
sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax[2], palette="magma")
ax[2].set_title("Top 15 Feature Importances")

plt.tight_layout()
plt.show()