import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from imapclient import IMAPClient
import email
from email.header import decode_header
import ssl
import getpass
import os
import time

MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# ----------------------------
# Model Training
# ----------------------------
def train_model(dataset_path):
    if not os.path.exists(dataset_path):
        print("File not found. Exiting...")
        return None, None

    data = pd.read_csv(dataset_path, encoding='latin-1')

    if "v1" in data.columns and "v2" in data.columns:
        data = data.rename(columns={"v1": "label", "v2": "text"})
    data = data[["label", "text"]]
    data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label_num'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(vectorizer.transform(X_test))
    print(f"Model trained with accuracy: {accuracy_score(y_test, preds):.2f}")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("Model and vectorizer saved successfully!")
    return model, vectorizer

# ----------------------------
# Load Model
# ----------------------------
def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        print("Model loaded successfully!")
        return model, vectorizer
    else:
        return None, None

# ----------------------------
# Gmail Classification
# ----------------------------
def classify_gmail(model, vectorizer):
    while True:
        EMAIL = input("Enter your Gmail address (or 'back' to return to main menu): ").strip()
        if EMAIL.lower() == 'back':
            break
        PASSWORD = getpass.getpass("Enter your Gmail App Password: ")

        folder = input("Enter Gmail folder to scan (e.g., INBOX, Promotions, Updates) or leave blank for INBOX: ").strip()
        folder = folder if folder else "INBOX"

        mark_action = input("Automatically mark spam as read/move to Spam folder? (yes/no): ").strip().lower()
        mark_spam = mark_action == "yes"

        try:
            context = ssl.create_default_context()
            with IMAPClient("imap.gmail.com", ssl_context=context) as imap_server:
                imap_server.login(EMAIL, PASSWORD)
                imap_server.select_folder(folder, readonly=False)

                messages = imap_server.search(['UNSEEN'])
                print(f"Found {len(messages)} unread emails in {folder}.\n")

                results = []

                for uid, message_data in imap_server.fetch(messages, ['BODY[]']).items():
                    raw_email = message_data[b'BODY[]']
                    msg = email.message_from_bytes(raw_email)

                    # Decode subject and sender
                    subject, encoding = decode_header(msg['Subject'])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else 'utf-8', errors='ignore')
                    from_ = msg.get('From')
                    date_ = msg.get('Date', 'Unknown')

                    # Extract plain text body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body += part.get_payload(decode=True).decode(errors='ignore')
                    else:
                        body = msg.get_payload(decode=True).decode(errors='ignore')

                    # Predict spam/ham
                    X_input = vectorizer.transform([body])
                    pred = model.predict(X_input)[0]
                    label = "Spam" if pred == 1 else "Ham"
                    print(f"\nFrom: {from_}\nSubject: {subject}\nDate: {date_}\nPrediction: {label}")

                    # Save results
                    results.append({
                        "From": from_,
                        "Subject": subject,
                        "Date": date_,
                        "Body": body,
                        "Prediction": label
                    })

                    # Mark spam emails
                    if mark_spam and label == "Spam":
                        imap_server.add_flags(uid, [b'\\Seen'])
                        try:
                            imap_server.move(uid, "[Gmail]/Spam")
                        except:
                            print("Could not move email to Spam folder. Ensure folder exists.")

                # Save results to CSV
                if results:
                    output_file = f"classified_emails_{EMAIL.replace('@','_').replace('.','_')}.csv"
                    df = pd.DataFrame(results)

                    # Add summary row
                    summary = {
                        "From": "",
                        "Subject": "",
                        "Date": "",
                        "Body": "Summary",
                        "Prediction": f"Total: {len(df)}, Ham: {sum(df['Prediction']=='Ham')}, Spam: {sum(df['Prediction']=='Spam')}"
                    }
                    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
                    df.to_csv(output_file, index=False)
                    print(f"\nAll classified emails saved to '{output_file}' with summary.")

                imap_server.logout()
        except Exception as e:
            print(f"Error connecting to Gmail: {e}")
            continue

# ----------------------------
# Automatic Gmail Scan (Scheduled every 300 minutes)
# ----------------------------
def classify_gmail_auto(model, vectorizer, interval_minutes=300):
    print(f"\nStarting automatic Gmail scans every {interval_minutes} minutes. Press Ctrl+C to stop.")
    try:
        while True:
            print(f"\nStarting Gmail scan at {pd.Timestamp.now()}")
            classify_gmail(model, vectorizer)
            print(f"\nWaiting {interval_minutes} minutes for next scan...")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nAutomatic scanning stopped by user.")

# ----------------------------
# Manual Message Testing
# ----------------------------
def manual_testing(model, vectorizer):
    while True:
        user_input = input("\nType a message to classify (or 'back' to return to main menu): ")
        if user_input.lower() == "back":
            break
        X_input = vectorizer.transform([user_input])
        pred = model.predict(X_input)[0]
        print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")

# ----------------------------
# Main Interactive Menu
# ----------------------------
model, vectorizer = load_model()

while True:
    print("\n--- Main Menu ---")
    print("1. Upload a dataset to train the spam model")
    print("2. Classify Gmail emails")
    print("3. Manual message testing")
    print("4. Exit")
    print("5. Automatic Gmail scans every 300 minutes")
    choice = input("Enter your choice (1-5): ").strip()

    if choice == "1":
        dataset_path = input("Enter full path to your CSV dataset: ").strip()
        model, vectorizer = train_model(dataset_path)
    elif choice == "2":
        if model is None or vectorizer is None:
            print("No trained model found. Please upload a dataset first.")
            dataset_path = input("Enter full path to your CSV dataset: ").strip()
            model, vectorizer = train_model(dataset_path)
        classify_gmail(model, vectorizer)
    elif choice == "3":
        if model is None or vectorizer is None:
            print("No trained model found. Please upload a dataset first.")
            dataset_path = input("Enter full path to your CSV dataset: ").strip()
            model, vectorizer = train_model(dataset_path)
        manual_testing(model, vectorizer)
    elif choice == "4":
        print("Exiting program.")
        break
    elif choice == "5":
        if model is None or vectorizer is None:
            print("No trained model found. Please upload a dataset first.")
            dataset_path = input("Enter full path to your CSV dataset: ").strip()
            model, vectorizer = train_model(dataset_path)
        classify_gmail_auto(model, vectorizer, interval_minutes=300)
    else:
        print("Invalid choice. Please enter 1-5.")
