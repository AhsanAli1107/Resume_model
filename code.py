# import labraries
import streamlit  as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu
import pickle
import joblib
from PyPDF2 import PdfReader
import docx
from docx import Document


st.title('Resume Screening App')


# Create a horizontal navigation menu
selected = option_menu(
    menu_title=None,  # No menu title for top navbar
    options=["Data + Visualization", "Prediction"],
    icons=["house", "bar-chart-line", "info-circle"],
    menu_icon="cast",  # Optional menu icon
    default_index=0,  # Default to the first option
    orientation="horizontal",
)

# Display the selected page
    
if selected == "Data + Visualization":
    st.title("This is where data will be displayed.")
    # load dataset
    df = pd.read_csv('/Users/ahsanali/Desktop/Resume_dataset/UpdatedResumeDataSet.csv')
    st.write(df.head())

    # EDA
    st.header('Visualization Cetagory Vise')
    counts = df['Category'].value_counts()
    labels = df['Category'].unique()
    # Plotting in Streamlit
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0, 1, len(labels))))
    st.pyplot(fig)

    # Data cleaning 
    def cleanResume(txt):
        cleantext = re.sub('http\S+\s', ' ',txt)
        cleantext = re.sub('RT|cc', ' ', cleantext)
        cleantext = re.sub('#\S+\s', ' ', cleantext)
        cleantext = re.sub('@\S+', ' ',cleantext)
        cleantext = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleantext)
        cleantext = re.sub(r'[^\x00-\x7f]', ' ', cleantext)
        cleantext = re.sub('\s+', ' ',cleantext)
        return(cleantext)

    df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))
    st.header("Cleaned Resume Text for Index 1:")
    st.write(df['Resume'][1])

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Fit and transform the 'Category' column
    le.fit(df['Category'])
    df['Category'] = le.transform(df['Category'])
    
    # Vactorization 
    tfidf = TfidfVectorizer(stop_words = 'english')
    tfidf.fit(df['Resume'])
    required_txt = tfidf.transform(df['Resume'])

    # Spliting the data
    X_train,X_test,y_train,y_test = train_test_split(required_txt, df['Category'], test_size=0.2, random_state=42)

    # Train the Model and Classification Report
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train,y_train)
    ypred = clf.predict(X_test)

    # print the  Accuracy
    st.title('Accuracy of the Model')
    st.header(accuracy_score(y_test, ypred))
    
elif selected == "Prediction":
    st.title("Information about Prediction.")

    # laod the model
    # Load pre-trained models
    tfidf = joblib.load("tfidf.pk1")
    clf = joblib.load("clf.pk2")

    # Function to clean the resume text
    def cleanResume(resume_text):
        # Add your resume cleaning logic here
        return resume_text  # Replace with actual cleaned text

    # Function to extract text from PDF
    def extract_text_from_pdf(file):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    # Function to extract text from Word documents
    def extract_text_from_docx(file):
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    # Category mapping
    Category_Mapping = {
        6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts', 24: 'Web Designing',
        16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness', 5: 'Civil Engineer',
        15: 'Java Developer', 4: 'Business Analyst', 21: 'SAP Developer', 2: 'Automation Testing',
        11: 'Electrical Engineering', 18: 'Operations Manager', 20: 'Python Developer', 
        8: 'DevOps Engineer', 17: 'Network Security Engineer', 19: 'PMO', 7: 'Database', 
        13: 'Hadoop', 10: 'ETL Developer', 9: 'DotNet Developer', 3: 'Blockchain', 23: 'Testing'
    }

    # Streamlit app
    st.title("Resume Classifier")

    # File uploader for PDF and Word documents
    uploaded_file = st.file_uploader("Upload your resume (PDF or Word)", type=["pdf", "docx"])

    # Text area for manual input
    manual_input = st.text_area("Or paste the resume text here:")

    # Prepare the input text
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(uploaded_file)
    else:
        resume_text = manual_input

    if st.button("Classify Resume"):
        if resume_text:
            # Clean the input resume
            cleaned_resume = cleanResume(resume_text)

            # Transform the cleaned resume using the pre-trained TfidfVectorizer
            input_features = tfidf.transform([cleaned_resume])

            # Make a prediction using the loaded classifier
            prediction_id = clf.predict(input_features)[0]

            # Map the predicted ID to the category name
            category_name = Category_Mapping.get(prediction_id, 'Unknown')
            
            # Display the result
            st.write(f"**Predicted Category:** {category_name}")
        else:
            st.warning("Please enter the resume text or upload a file to classify.")