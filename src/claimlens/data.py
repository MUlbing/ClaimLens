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
# Step 1. Definitions for extracting and cleaning the text
# ---------------------------------------------------------------------------

# Greetings that are removed from the chat
GREETING_RE = re.compile(
    r"\b(?:hi+|ha+i+|hello+|hellow+|helo+|herro+|hallo|hey+|heya|hiya|howdy|hola|greetings|yo|ho|"
    r"what(?:'s| is|s) up(?![\s,]*[A-Za-z])|sup|"
    r"good (?:morning|afternoon|evening)|how are you(?: doing)?|how(?:'s| is) it going|"
    r"are you (?:still )?there|(?:is )?(?:anyone|anybody) (?:still )?(?:there|here|home))\b"
    r"(?:\s+there)?[\s.,!?:;()\-]*",
    re.IGNORECASE,
)

# Names: removed from the text (after "my name is" / "I am", or a message that is only a name)
NAME_INTRO_RE = re.compile(r"\b(?:my (?:name|anme)(?: is|'s)|name is)\s+[a-z]+(?:\s+[a-z]+)?", re.IGNORECASE)
NAME_CAPS_RE = re.compile(r"(?i:\b(?:i am|i'm|im|this is))\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
BARE_NAME_RE = re.compile(r"^\W*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\W*$")

# Confirmations and politeness: removed in CleanText and ignored when looking for intent
EXTRA_STOP_WORDS = {
    "yes", "yeah", "yep", "yup", "nope", "ok", "okay", "sure", "hmm", "hmmm", "ah", "ahh", "oh", "ohh", "wow",
    "thanks", "thank", "please", "sorry", "today", "well", "fine", "doing",
    "actually", "really", "just", "ask", "asking", "talk", "speak", "u", "r", "ur", "there",
}

# Words without intent: only used to decide when to stop, they are NOT removed from the text
NO_INTENT = (set(ENGLISH_STOP_WORDS) - {"bill", "back", "amount", "call", "interest", "system", "move", "full"}) | EXTRA_STOP_WORDS | {
    # announcements without content
    "question", "questions", "query", "queries", "problem", "problems", "issue", "issues",
    "complaint", "concern", "help", "helping", "assist", "assistance", "connected",
    "going", "welcome", "need", "want", "wanted", "like", "looking", "wondering",
    "general", "simple", "little", "quick", "answer", "info", "information", "bad", "favor", "hope",
    "good", "great",
}

# Stop words that are NOT removed in CleanText, because they carry signal
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

# Words removed in CleanText: sklearn stop words minus KEEP_WORDS, plus the extra words
STOP_WORDS = (set(ENGLISH_STOP_WORDS) - KEEP_WORDS) | EXTRA_STOP_WORDS

# ----------------------------------------------------------------------------------------------------
# Step 1b. Definitions for the labels (y)
# ----------------------------------------------------------------------------------------------------

# Subflows treated as refund intent. mistimed_billing_never_bought was added because 67% of its chats end with a money action.
REFUND_INTENT_SUBFLOWS = {
    "refund_initiate", "refund_update", "cost", "manage_dispute_bill", "manage_cancel",
    "status_service_added", "status_quantity", "return_size", "return_color", "return_stain",
    "status_mystery_fee", "mistimed_billing_never_bought",
}

# Subflows where a money action counts as a return. In any other subflow it counts as a refund.
RETURN_SUBFLOWS = {"return_size", "return_color", "return_stain"}

# Action text that tells a refund was issued
REFUND_TEXT = "refund has been made"
# Fee-removal action: "Order has been updated with 64." or "$12.95."
REFUND_FEE = re.compile(
    r"^\s*order has been updated with\s+\$?\d{1,3}(?:\.\d{1,2})?\.\s*$",
    re.IGNORECASE,
)
# Other money actions: credit, fee waiver, cancellation and the return method (mail, in store, drop off)
MONEY_ACTION = re.compile(
    r"^\s*order has been updated with\s*(give credit|waive fee|cancel (order|shipment)|(by )?mail|in store|drop ?off)",
    re.IGNORECASE,
)
def is_money_action(text):
    """Return True if the action text means a refund, a removed fee, a credit, a cancellation or a return."""
    return REFUND_TEXT in text.lower() or bool(REFUND_FEE.search(text)) or bool(MONEY_ACTION.search(text))




# ----------------------------------------------------------------------------------------------------
# Step 2. Extract messages from client from the raw_data and create the dataframe
# Class ChatData: first messages (X for model 1), all customer messages (X for model 2) and the labels
# ----------------------------------------------------------------------------------------------------

class ChatData:
    """Goes from the raw data to one dataframe: first_msgs, full_msgs and the labels refund_intent, intent_approved, approved_type."""

    def extract_first(self, turns):
        """Customer messages before the first agent reply that comes after real content.
        Agent turns are ignored until the customer wrote something with intent.
        Names are removed: after "my name is" or "I am", and messages that are only a name."""
        parts, has_content = [], False
        for speaker, text in turns:
            if speaker == "customer":
                # Straight apostrophes, no greetings, contractions expanded
                text = GREETING_RE.sub(" ", text.replace("’", "'")).strip()
                text = contractions.fix(text)
                if BARE_NAME_RE.match(GREETING_RE.sub(" ", text).strip()):
                    continue  # the message is only a name
                text = NAME_CAPS_RE.sub(" ", NAME_INTRO_RE.sub(" ", text))  # names after "my name is" / "I am"
                if re.search(r"[A-Za-z0-9]", text):
                    parts.append(text)
                # Does the message have words with intent? (not a greeting or filler)
                check = GREETING_RE.sub(" ", text).lower().replace("'", "")
                check = re.sub(r"(.)\1{2,}", r"\1", check)  # helllllp -> help, sooooo -> so
                if any(len(w) > 1 and w not in NO_INTENT for w in re.findall(r"[a-z]+", check)):
                    has_content = True
            elif speaker == "agent" and has_content:
                break
        return " ".join(" ".join(parts).split())

    def extract_full(self, turns):
        """All customer messages with the same cleanup, without stopping at agent replies."""
        parts = []

        for speaker, text in turns:
            if speaker != "customer":
                continue

            text = GREETING_RE.sub(" ", text.replace("’", "'")).strip()
            text = contractions.fix(text)

            if BARE_NAME_RE.match(GREETING_RE.sub(" ", text).strip()):
                continue

            text = NAME_CAPS_RE.sub(" ", NAME_INTRO_RE.sub(" ", text))

            if re.search(r"[A-Za-z0-9]", text):
                parts.append(text)

        return " ".join(" ".join(parts).split())

    def build_df(self, data):
        """One row per chat: split, id, flow, subflow, first_msgs, full_msgs, refund_intent, intent_approved, approved_type.

        intent_approved: 1 if the chat has a money action (refund, fee removed, credit, cancellation, return), else 0.
        refund_intent: 1 if the subflow is in REFUND_INTENT_SUBFLOWS, or if intent_approved is 1 (a chat with a money action has the intent).
        approved_type: "return" if intent_approved is 1 and the subflow is a return subflow, "refund" for any other subflow, 0 if not approved."""
        rows = []
        for split in ("train", "dev", "test"):
            for chat in data[split]:
                turns, scenario = chat["original"], chat.get("scenario", {})
                subflow = scenario.get("subflow")
                approved = any(speaker == "action" and is_money_action(text) for speaker, text in turns)
                if approved:
                    approved_type = "return" if subflow in RETURN_SUBFLOWS else "refund"
                else:
                    approved_type = 0
                rows.append({
                    "split": split,
                    "id": chat["convo_id"],
                    "flow": scenario.get("flow"),
                    "subflow": subflow,
                    "first_msgs": self.extract_first(turns),
                    "full_msgs": self.extract_full(turns),
                    "refund_intent": int(subflow in REFUND_INTENT_SUBFLOWS or approved),  # label for model 1
                    "intent_approved": int(approved),  # label for model 2
                    "approved_type": approved_type,  # "refund", "return" or 0
                })
        return pd.DataFrame(rows)

# ----------------------------------------------------------------------------------------------------
# Step 4. Clean Client input (either from first message or API input)
# ----------------------------------------------------------------------------------------------------

class CleanText:
    """Cleans one text. Used on first_msgs and on any new customer message that arrives by API."""

    def __init__(self, remove_stop_words=True, lemmatize=True):
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        """Eliminates Greetings, names, punctuation, numbers and stop words. Lowercase and lemmas. Returns a string."""
        text = GREETING_RE.sub(" ", text.replace("’", "'")).strip()  # greetings go before the contractions
        text = contractions.fix(text)
        if BARE_NAME_RE.match(text):
            return ""  # the message is only a name
        text = NAME_CAPS_RE.sub(" ", NAME_INTRO_RE.sub(" ", text))  # names go before .lower(), they need capitals
        text = text.lower().replace("'", "")           # john's -> johns
        text = re.sub(r"[^\w\s]", " ", text)           # other punctuation -> space (t-shirt -> t shirt)
        text = re.sub(r"\b\w*\d\w*\b", " ", text)      # words with a digit are removed (49, 1st, abc123)
        words = text.split()
        if self.remove_stop_words:
            words = [w for w in words if w not in STOP_WORDS]
        if self.lemmatize:  # verb first, then noun
            words = [self.lemmatizer.lemmatize(self.lemmatizer.lemmatize(w, pos="v"), pos="n") for w in words]
        return " ".join(words)
