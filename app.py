from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def recommendation(age, disease, gender):
    medi_data = pd.read_csv("./webmd.csv")
    medi_data = medi_data[~medi_data.isin([' '])]  # remove missing
    medi_data = medi_data.dropna(axis=0)  # drop null values
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'Sex'.
    medi_data['Sex'] = label_encoder.fit_transform(medi_data['Sex'])

    medi_data['Condition'] = medi_data['Condition'].str.lower()
    medi_data['Reviews'] = medi_data['Reviews'].str.lower()

    disease = disease.lower()
    gender = gender.lower()
    if gender == "female" or gender == "f":
        gender = 0
    else:
        gender = 1
    filtered_users = []
    # similar disease users
    for i in range(len(medi_data)):
        if disease in medi_data.iloc[i]["Condition"]:
            filtered_users.append(medi_data.iloc[i])
    filtered_user_data = pd.DataFrame(filtered_users)
    # similar age and gender users
    filtered_users = []
    for i in range(len(filtered_user_data)):
        ag = filtered_user_data.iloc[i]['Age']
        if "or" in ag:
            ag = ag.split(" ")
            ag1 = int(ag[0])
            ag2 = 100
        elif "-" in ag:
            ag = ag.split("-")
            if len(ag[-1]) > 3:
                ag1 = 0
                ag2 = 1
            else:
                ag1 = int(ag[0])
                ag2 = int(ag[1])
        if age >= ag1 and age <= ag2:
            if filtered_user_data.iloc[i]['Sex'] == gender:
                filtered_users.append(filtered_user_data.iloc[i])
    filtered_userframe = pd.DataFrame(filtered_users)

    filtered_userframe.loc[(filtered_userframe['Satisfaction'] >= 3), 'Review_Sentiment'] = 1
    filtered_userframe.loc[(filtered_userframe['Satisfaction'] < 3), 'Review_Sentiment'] = 0

    # to get  CV and tfID vectors for review analysis
    def tfIDfV(review):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(review)
        X = X.toarray()
        return X

    data = tfIDfV(filtered_userframe["Reviews"])
    labels = to_categorical(filtered_userframe["Review_Sentiment"], num_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    def createModel():
        model = tf.keras.Sequential()
        model.add(layers.Dense(25, input_shape=(x_train.shape[1],)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(25))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(25))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(2, activation='softmax'))
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
        model.summary()
        return model

    model = createModel()
    hist = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
    ls = []
    preds = model.predict(data)
    preds = np.argmax(preds, axis=1)
    for i in preds:
        if i == 1:
            ls.append("positive")
        else:
            ls.append("negative")
    filtered_userframe['Comments'] = ls
    filtered_userframe1 = filtered_userframe[filtered_userframe["Comments"] == "positive"]
    from textblob import TextBlob
    reviews = filtered_userframe1['Reviews']
    Predict_Sentiment = []
    for review in (reviews):
        blob = TextBlob(review)
        Predict_Sentiment += [blob.sentiment.polarity]
    filtered_userframe1["Predict_Sentiment"] = Predict_Sentiment
    filtered_userframe1 = filtered_userframe1[filtered_userframe1["Predict_Sentiment"] > 0]
    drug = {}
    for i in range(len(filtered_userframe1)):
        x = filtered_userframe1.iloc[i]
        if x['DrugId'] in drug.keys():
            drug[x['DrugId']][2].append(
                (x['EaseofUse'] + x['Effectiveness'] + x['Satisfaction'] + x['UsefulCount']) / 4)
        else:
            drug[x['DrugId']] = [x['DrugId'], x['Drug'],
                                 [(x['EaseofUse'] + x['Effectiveness'] + x['Satisfaction'] + x['UsefulCount']) / 4]]
    for i in drug.keys():
        drug[i][2] = sum(drug[i][2]) / len(drug[i][2])
        drug[i][2] = round(drug[i][2], 2)
    drug_data = pd.DataFrame(drug.values())
    drug_data.columns = ['DrugID', 'Drug_name', 'Mean_Rating']
    drug_data = drug_data.sort_values(by=['Mean_Rating'], ascending=False)
    return drug_data


app = Flask(__name__)

disease_pred_model = pickle.load(open('smote_xgboost_model.pkl', 'rb'))


# medicine_pred_model = pickle.load(open('finalized_detection_model.pkl','rb'))


@app.route('/', methods=['POST'])
def home():
    return ()


@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    q1 = (request.form.get('symptoms')).split(',')
    print(q1)
    query1 = []
    c= 0
    data_sym = pd.read_csv("./Symptom-severity.csv")
    data_dis = pd.read_csv("./Disease_Label.csv")

    for i in range(len(q1)):
        for j in range(len(data_sym)):
            if q1[i] in data_sym.iloc[j]["Symptom"]:
                temp = data_sym.iloc[j]["weight"]
                query1.append(temp)
                c+=1
                break
    for j in range(17 - c):
        query1.append(0)

    # q2 = (request.form.get('disease')).split(',')
    # query2 = [float(x) for x in q2]
    input_query1 = np.array([query1])
    print(input_query1)
    # input_query2 = np.array([query2])
    result1 = (disease_pred_model.predict(input_query1))
    # result2 = (medicine_pred_model.predict(input_query2))
    for k in range(len(data_dis)):
        if result1[0] == data_dis.iloc[k]["Disease_Labels"]:
            d = data_dis.iloc[k]["Disease"]
            break
    return jsonify({"disease": str(d)})


if __name__ == '__main__':
    app.run(debug=True)

#postgres://disxdrug_user:3InclU5eNJqMLBkPELK0iimVTEFVAr0A@dpg-cgmj2l3k9u59cruuprh0-a.oregon-postgres.render.com/disxdrug