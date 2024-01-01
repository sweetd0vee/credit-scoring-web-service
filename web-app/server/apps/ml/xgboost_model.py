import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class XGBoostClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        df = pd.read_csv(path_to_artifacts + 'data/train.csv')
        self.train = df.set_index('SK_ID_CURR')
        self.values_fill_missing = joblib.load(path_to_artifacts + "train_mode.joblib")
        self.label_encoders = joblib.load(path_to_artifacts + "label_encoders.joblib")
        self.target_encoders = joblib.load(path_to_artifacts + "label_encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "XGBoost.joblib")

    def convert_input_dict(self, temp):
        features = ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                    'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START',
                    'HOUR_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1',
                    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'LANDAREA_AVG', 'APARTMENTS_MODE',
                    'YEARS_BEGINEXPLUATATION_MEDI', 'DAYS_LAST_PHONE_CHANGE',
                    'FLAG_DOCUMENT_3', 'b_closed_Consumer credit_num',
                    'b_active_all_num', 'b_Consumer credit_sum_1', 'b_all_sum_1',
                    'b_Credit card_sum_3']
        temp_row = []
        for col in features:
            if col in temp and temp[col] != "":
                temp_row.append(temp[col])
            else:
                # fill missing values
                temp_row.append(self.values_fill_missing[col])
        input_data = pd.DataFrame([temp_row], columns=features)
        return input_data


    def preprocessing(self, input_data):
        categorical = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                       'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FLAG_DOCUMENT_3']


        # convert categoricals
        for col in categorical:
            le = self.label_encoders[col]
            input_data[col] = le.transform(input_data[col])
            TE = self.target_encoders[col]
            input_data[col] = TE.transform(input_data[col])


        features_from_train = ['b_Consumer credit_sum_1', 'b_Credit card_sum_3', 'b_active_all_num',
                               'b_all_sum_1', 'b_closed_Consumer credit_num']
        # extract features from train
        id = input_data['SK_ID_CURR']
        idx = self.train.index
        if sum(idx.isin([id])) == 1:
            for col in features_from_train:
                input_data[col] = self.train.loc[id][col]
        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)[:,1]

    def postprocessing(self, preds):
        label = 0
        if preds[0] > 0.5:
            label = 1
        return {"probability": preds[0], "label": label, "status": "OK"}

    def compute_prediction(self, temp):
        try:
            input_data = self.convert_input_dict(temp)
            input_data = self.preprocessing(input_data)

            # features = ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
            #             'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            #             'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
            #             'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
            #             'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START',
            #             'HOUR_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1',
            #             'EXT_SOURCE_2', 'EXT_SOURCE_3', 'LANDAREA_AVG', 'APARTMENTS_MODE',
            #             'YEARS_BEGINEXPLUATATION_MEDI', 'DAYS_LAST_PHONE_CHANGE',
            #             'FLAG_DOCUMENT_3', 'b_closed_Consumer credit_num',
            #             'b_active_all_num', 'b_Consumer credit_sum_1', 'b_all_sum_1',
            #             'b_Credit card_sum_3']
            # test_values = [0.00000000e+00, 2.47500000e+05, 4.50000000e+05, 2.73240000e+04, 7.00000000e+00, 1.00000000e+00,
            #           2.00000000e+00, 9.17500000e-03,
            #           -1.34800000e+04, -3.00900000e+03, -4.50700000e+03, -4.32300000e+03,
            #           1.20718630e+01, 1.20000000e+01, 5.00000000e+00, 1.50000000e+01,
            #           3.00000000e+01, 5.01821414e-01, 7.45130792e-01, 5.10934269e-01,
            #           6.63607545e-02, 1.14351576e-01, 9.77665703e-01, -9.70000000e+02,
            #           1.00000000e+00, 3.01455502e+00, 2.05673734e+00, 2.92859207e+05,
            #           4.54251327e+05, 1.48118778e+05]
            #input_data = pd.DataFrame([test_values], columns=features)
  
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
