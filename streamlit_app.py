import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import io

st.title("Fruit Dataset Analysis and Classification")

# Upload file
uploaded_file = st.file_uploader(
    "Upload your fruit dataset (Excel file):", type=["xlsx"])

if uploaded_file:
    dsfruit = pd.read_excel(uploaded_file)

    st.header("Dataset Overview")
    st.write("Shape of the dataset:", dsfruit.shape)
    st.write("First 5 rows:")
    st.dataframe(dsfruit.head())

    st.write("Last 5 rows:")
    st.dataframe(dsfruit.tail())

    st.write("Dataset description:")
    st.dataframe(dsfruit.describe())

    st.write("Dataset information:")
    buffer = io.StringIO()
    dsfruit.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("Missing values:")
    st.write(dsfruit.isnull().sum())

    # Form to add new data
    st.subheader("Add New Data")
    with st.form("add_data_form"):
        new_data = {}
        for col in dsfruit.columns:
            new_data[col] = st.text_input(f"Enter {col}:")
        submitted = st.form_submit_button("Add Data")

        if submitted:
            try:
                new_row = pd.DataFrame([new_data])
                dsfruit = pd.concat([dsfruit, new_row], ignore_index=True)
                st.success("New data added successfully!")
                st.dataframe(dsfruit.tail())
            except Exception as e:
                st.error(f"Error adding data: {e}")

    # Encode categorical column
    le = LabelEncoder()
    dsfruit['name'] = le.fit_transform(dsfruit['name'])
    st.write("Transformed dataset:")
    st.dataframe(dsfruit)

    # Count plot
    st.subheader("Name Variable Count Plot")
    fig, ax = plt.subplots()
    sns.countplot(dsfruit["name"], ax=ax)
    ax.set_xlabel("Name")
    ax.set_ylabel("Count of Name")
    ax.set_title("Name variable count")
    st.pyplot(fig)

    # Features and target
    x = dsfruit.iloc[:, :-1]
    y = dsfruit.iloc[:, -1]

    st.write("Features shape:", x.shape)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0)

    # Random Forest Classifier
    classifier = RandomForestClassifier(
        criterion='gini', max_depth=8, min_samples_split=10, random_state=0)
    classifier.fit(x_train, y_train)

    # Feature Importances
    st.subheader("Feature Importances")
    importances = classifier.feature_importances_
    indices = np.argsort(importances)
    features = dsfruit.columns[:-1]

    fig, ax = plt.subplots()
    ax.barh(range(len(indices)),
            importances[indices], color='b', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importances')
    st.pyplot(fig)

    # Predictions
    y_pred = classifier.predict(x_test)

    # Metrics
    st.subheader("Model Evaluation")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(classifier, x_train, y_train, cv=10)
    st.write("Cross-validation scores:", cv_scores)
    st.write("Mean cross-validation score:", np.mean(cv_scores))

else:
    st.info("Awaiting for Excel file upload. Please upload a file to proceed.")
