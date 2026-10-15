"""ClaimLens text classes."""

import json
import re
from pathlib import Path

import contractions
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ---------------------------------------------------------------------------
# Step 0. Function to import raw data
# ---------------------------------------------------------------------------
def load_raw_data(path):
    """Load the ABCD json as a dict: {"train": [...], "dev": [...], "test": [...]}."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1. FirstMessage (notebook 1)
# ---------------------------------------------------------------------------
class FirstMessage:
    """Extracts the customer's first messages from a chat (the X of model 1). Uses CleanText for greetings and contractions."""

    # Names: only used to decide if a message has content, the text is not modified
    NAME_INTRO_RE = re.compile(r"\b(?:my (?:name|anme)(?: is|'s)|name is)\s+[a-z]+(?:\s+[a-z]+)?", re.IGNORECASE)
    NAME_CAPS_RE = re.compile(r"(?i:\b(?:i am|i'm|im|this is))\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
    BARE_NAME_RE = re.compile(r"^\W*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\W*$")

    # Words without intent: only used to decide when to stop, they are NOT removed from the text
    NO_INTENT = (set(ENGLISH_STOP_WORDS) - {"bill", "back", "amount", "call", "interest", "system", "move", "full"}) | {
        # announcements without content
        "question", "questions", "query", "queries", "problem", "problems", "issue", "issues",
        "complaint", "concern", "help", "helping", "assist", "assistance", "connected",
        "going", "welcome", "need", "want", "wanted", "like", "looking", "wondering",
        "general", "simple", "little", "quick", "answer", "info", "information", "bad", "favor", "hope",
        # confirmations and politeness (not in the sklearn stop list)
        "yes", "yeah", "yep", "yup", "nope", "ok", "okay", "sure", "hmm", "hmmm", "ah", "ahh", "oh", "ohh", "wow",
        "thanks", "thank", "please", "sorry", "today", "well", "good", "great", "fine", "doing",
        "actually", "really", "just", "ask", "asking", "talk", "speak", "u", "r", "ur", "there",
    }


    def __init__(self, cleaner=None):
        self.cleaner = cleaner or CleanText()

    def intent_words(self, text):
        """Words with intent. Empty if the message is only a greeting, filler or a name."""
        text = self.cleaner.remove_greetings(text)
        if self.BARE_NAME_RE.match(text):
            return []
        text = self.NAME_CAPS_RE.sub(" ", self.NAME_INTRO_RE.sub(" ", text)).lower().replace("'", "")
        text = re.sub(r"(.)\1{2,}", r"\1", text)  # helllllp -> help, sooooo -> so
        return [w for w in re.findall(r"[a-z]+", text) if len(w) > 1 and w not in self.NO_INTENT]

    def extract(self, turns):
        """Agent turns are ignored until the customer wrote something with intent.
        After that, the first agent reply ends the extraction. Stop words are kept."""
        parts, has_content = [], False
        for speaker, text in turns:
            if speaker == "customer":
                text = self.cleaner.normalize(text)
                if re.search(r"[A-Za-z0-9]", text):
                    parts.append(text)
                if self.intent_words(text):
                    has_content = True
            elif speaker == "agent" and has_content:
                break
        return " ".join(" ".join(parts).split())


    def build_df(self, data):
        """One row per chat: split, id, flow, subflow, first_msgs. No labels."""
        rows = []
        for split in ("train", "dev", "test"):
            for chat in data[split]:
                scenario = chat.get("scenario", {})
                turns = chat["original"]
                rows.append({
                    "split": split,
                    "id": chat["convo_id"],
                    "flow": scenario.get("flow"),
                    "subflow": scenario.get("subflow"),
                    "first_msgs": self.extract(turns),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2. BuildingDataFrame (notebook 2)
# ---------------------------------------------------------------------------
class BuildingDataFrame:
    """Adds the two labels to the first df from notebook 1. Keeps all its columns."""

    # Action text that tells a refund was issued (used for the action column)
    REFUND_TEXT = "refund has been made"

    # v1 list of subflows treated as refund intent. Not validated yet.
    REFUND_INTENT_SUBFLOWS = {
        "refund_initiate", "refund_update", "cost", "manage_dispute_bill", "manage_cancel",
        "status_service_added", "status_quantity", "return_size", "return_color", "return_stain",
        "bad_price_competitor", "status_mystery_fee", "manage",
    }
    # Constants: REFUND_INTENT_SUBFLOWS

    def refund_intent(self, subflow):
        """Label for model 1: 1 if the subflow is in the refund intent list."""
        return int(subflow in self.REFUND_INTENT_SUBFLOWS)

    def refund_output(self, turns):
        """Label for model 2: 1 if the chat has the action 'A refund has been made...', else 0."""
        return int(any(speaker == "action" and self.REFUND_TEXT in text.lower() for speaker, text in turns))

    def refund_output_df(self, data):
        """refund_output for every chat, with split and id to join it to the first df."""
        rows = []
        for split in ("train", "dev", "test"):
            for chat in data[split]:
                rows.append({
                    "split": split,
                    "id": chat["convo_id"],
                    "refund_output": self.refund_output(chat["original"]),
                })
        return pd.DataFrame(rows)

    def build(self, first_message_df, data):
        """Adds refund_intent, refund_output and n_words. It needs the raw data to read the actions."""
        df = first_message_df.copy()
        df["refund_intent"] = df["subflow"].map(self.refund_intent)
        df = df.merge(self.refund_output_df(data), on=["split", "id"], how="left", validate="one_to_one")
        df["n_words"] = df["first_msgs"].str.split().str.len()
        return df


# ---------------------------------------------------------------------------
# Step 3. CleanText (notebook 3). Also used on messages that arrive by API
# ---------------------------------------------------------------------------
class CleanText:
    """Cleans text. Used on first_msgs and on any new customer message (API). No tokenizing: the output is a string."""

    # Greetings that are removed from the text
    GREETING_RE = re.compile(
        r"\b(?:hi+|ha+i+|hello+|hellow+|helo+|herro+|hallo|hey+|heya|hiya|howdy|hola|greetings|yo|ho|"
        r"what(?:'s| is|s) up(?![\s,]*[A-Za-z])|sup|"
        r"good (?:morning|afternoon|evening)|how are you(?: doing)?|how(?:'s| is) it going|"
        r"are you (?:still )?there|(?:is )?(?:anyone|anybody) (?:still )?(?:there|here|home))\b"
        r"(?:\s+there)?[\s.,!?:;()\-]*",
        re.IGNORECASE,
    )

    # Stop words that are NOT removed, because they carry signal
    KEEP_WORDS = {
        # negations
        "no", "not", "nor", "never", "cannot",
        # words that separate refund intent from other intents
        "only", "too", "other", "another", "same",
        # money and service words that sklearn counts as stop words
        "bill", "amount", "interest", "system", "full", "call", "move", "back",
        # verbs from the v2 list
        "get", "give", "go", "keep", "put", "take",
    }
    STOP_WORDS = ENGLISH_STOP_WORDS - KEEP_WORDS

    def __init__(self, remove_stop_words=True, lemmatize=True):
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()

    # Steps shared with FirstMessage
    def remove_greetings(self, text):
        """Remove greeting words (and the punctuation after them)."""
        return self.GREETING_RE.sub(" ", text).strip()

    def normalize(self, text):
        """Straight apostrophes, greetings removed, contractions expanded (can't -> cannot).
        Greetings go first, because contractions.fix turns 'howdy' into 'how do you'."""
        text = text.replace("’", "'")
        text = self.remove_greetings(text)
        return contractions.fix(text)

    # Steps only in CleanText
    def lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        """Apostrophes are deleted (john's -> johns). Other punctuation becomes a space (t-shirt -> t shirt)."""
        text = text.replace("'", "").replace("’", "")
        return re.sub(r"[^\w\s]", " ", text)

    def remove_numbers(self, text):
        """Removes every word that has a digit (49, 1st, abc123)."""
        return re.sub(r"\b\w*\d\w*\b", " ", text)

    def drop_stop_words(self, text):
        """Removes the stop words, except the ones in KEEP_WORDS."""
        return " ".join(w for w in text.split() if w not in self.STOP_WORDS)

    def lemmatize_text(self, text):
        """Verb first, then noun."""
        words = text.split()
        return " ".join(self.lemmatizer.lemmatize(self.lemmatizer.lemmatize(w, pos="v"), pos="n") for w in words)

    def clean(self, text):
        text = self.normalize(text)
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        if self.remove_stop_words:
            text = self.drop_stop_words(text)
        if self.lemmatize:
            text = self.lemmatize_text(text)
        return " ".join(text.split())

    def tokenize(self, text):
        """Splits a text into a list of words. Use it on the output of clean."""
        return text.split()

    def build(self, df):
        """Keeps every column and adds clean_text (a string) and tokens (a list of words)."""
        df = df.copy()
        df["clean_text"] = df["first_msgs"].map(self.clean)
        df["tokens"] = df["clean_text"].map(self.tokenize)
        return df
