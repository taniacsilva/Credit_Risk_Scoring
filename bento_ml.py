# Import Libraries
import bentoml
from XGBoost_model import *

# XGBoost Model
eta = 0.1
md = 3 
ch = 6

args = parse_arguments()
data = data_preparation(args.file_name)
set_used = split_train_val_test(data)
X_full_train, X_test, dv = one_hot_enconding(set_used[6], set_used[2])
features = dv.get_feature_names_out()
model, y_pred, auc, output_string, scores, key_eta, key_md, key_ch =  xgb_model(X_full_train, set_used[7], features, X_test, set_used[5], eta, md, ch)

# Bento ML
bentoml.xgboost.save_model('credit_risk_model',
                            model,
                            custom_objects={'DictVectorizer': dv}, signatures={ # model signatures for runner info
                            "predict": {"batchable": True,
                                        "batch_dim": 0}})

# Model Version
print(bentoml.xgboost.save_model('credit_risk_model',
                            model,
                            custom_objects={'DictVectorizer': dv}, signatures={ # model signatures for runner info
                            "predict": {"batchable": True,
                                        "batch_dim": 0}}))
