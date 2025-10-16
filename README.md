Spam Email Detection Project – Summary
1. Purpose

Automatically detect whether an email is Spam or Ham (not spam).

Works with user-provided datasets or Gmail accounts.

Supports manual testing and automatic scanning.

2. Components
a. Dataset

CSV file with at least two columns:

label → “ham” or “spam”

text → email content

Can be uploaded by the user to train the model.

Multiple Gmail accounts generate separate CSV files.

b. Model

TF-IDF Vectorizer → converts text to numerical features.

Multinomial Naive Bayes → classifies email as Spam or Ham.

Model and vectorizer are saved using joblib for reuse.

c. Gmail Integration

Connects via IMAP and SSL.

Fetches unread emails.

Extracts From, Subject, Date, Body.

Classifies using the trained model.

Saves results to a CSV file with a summary of Ham/Spam counts.

Optionally, can mark Spam emails as read or move them to Spam folder.

d. Manual Testing

User can type any email content.

Model predicts Spam or Ham immediately.

e. Automatic Scanning

Optional feature to run every 300 minutes (5 hours).

Keeps the inbox monitored continuously.

Requires the script to remain running in the terminal.

3. Output

Separate CSV file per Gmail account:

classified_emails_<email>.csv


Contains all processed emails and classification results.

CSVs are saved in the project folder.

4. User Interaction

Script provides menu options:

Upload dataset to train model

Provide Gmail account to classify emails

Manual testing

Exit

Automatic scanning (optional)

5. Key Notes

The project is original; your implementation is unique.

The TF-IDF + Naive Bayes method is standard, but your integration with Gmail, CSV handling, and automatic scanning is your contribution.

For IEEE or papers: focus on your implementation details, results, and evaluation, written in your own words.

6. How to Run

Run the script:

python spam-email-detection.py


Follow menu options.

For Gmail classification, input your email and App Password.

CSVs will be saved in the project folder.