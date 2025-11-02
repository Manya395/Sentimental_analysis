from googleapiclient.discovery import build
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

nltk.download("stopwords")

# ---- Your real API key here ----
API_KEY = "Your google API Key"

youtube = build("youtube", "v3", developerKey=API_KEY)

# ---------- CLEAN ----------
def clean_comment(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------- SENTIMENT ----------
def get_sentiment(text):
    blob = TextBlob(text)
    p = blob.sentiment.polarity
    if p > 0: return "Positive"
    elif p < 0: return "Negative"
    else: return "Neutral"

# ---------- FETCH COMMENTS ----------
def get_comments(video_id):
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=20,
        textFormat="plainText"
    )
    res = req.execute()
    comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in res["items"]]
    return comments

# ---------- SAFE “FAKE RDD” ----------
class FakeRDD:
    def __init__(self, data): self.data = data
    def map(self, func): return FakeRDD(list(map(func, self.data)))
    def collect(self): return self.data

# ---------- MAIN ----------
if __name__ == "__main__":
    video_input = input("Enter YouTube video ID: ")

    # Extract ID from full URL or direct ID
    import re
    m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_input)
    video_id = m.group(1) if m else video_input.strip()

    print("\nFetching comments...")
    comments = get_comments(video_id)
    print(f"Fetched {len(comments)} comments!\n")

    # --- RDD-style pipeline ---
    comments_rdd = FakeRDD(comments)
    cleaned_rdd = comments_rdd.map(clean_comment)
    cleaned_comments = cleaned_rdd.collect()

    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}

    print("Analyzing Sentiments:\n")
    for orig, clean in zip(comments, cleaned_comments):
        s = get_sentiment(clean)
        sentiments[s] += 1
        print(f"Comment: {orig}\n→ Cleaned: {clean}\n→ Sentiment: {s}\n")

    # --- Visualization ---
    labels, values = sentiments.keys(), sentiments.values()
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("YouTube Comment Sentiment Distribution (RDD-style)")
    plt.show()

