"""Text cleaning and the 12 features shared by training and the Streamlit app."""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --------------------------------------------------------------------------- #
# NLTK bootstrap
# --------------------------------------------------------------------------- #
_NLTK_PACKAGES = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]


def ensure_nltk(quiet=True):
    """Download the NLTK corpora the feature extractor needs (idempotent)."""
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.download(pkg, quiet=quiet)
        except Exception:
            # Feature extraction can still continue when the corpora are already cached.
            pass


_LEMMATIZER = WordNetLemmatizer()

# --------------------------------------------------------------------------- #
# Both feature generation and the app import these keyword lists.
# --------------------------------------------------------------------------- #
URGENCY_KW = [
    r"hurry", r"limited time", r"ends in", r"only.*hours", r"today only",
    r"flash sale", r"act now", r"last chance", r"don't miss", r"expires",
    r"clock is ticking", r"ending soon", r"deal ends", r"final hours",
    r"sale ends", r"countdown", r"before it'?s gone",
]
SCARCITY_KW = [
    r"only \d+ left", r"low stock", r"selling fast", r"almost gone",
    r"high demand", r"\d+ remaining", r"last remaining", r"few left",
    r"running out", r"limited stock", r"\d+ in stock", r"nearly sold out",
]
SHAME_PHRASES = [
    r"no thanks.*don't want", r"i don't need", r"no.*i prefer.*paying",
    r"no.*i hate", r"no thanks", r"prefer to pay", r"hate saving",
    r"dont want", r"i'?ll pay full", r"i don'?t like saving", r"no, i'?m fine paying",
]
CANCEL_DIFF = [
    r"to cancel.*call", r"email.*unsubscribe", r"to stop.*write",
    r"cancel.*call.*1-800", r"registered mail", r"hotline",
    r"loyalty department", r"chat with our live agent", r"cancellation fee",
    r"call.*to cancel", r"business hours", r"retention team",
]
SOCIAL_PROOF = [
    r"people.*viewing", r"people.*bought", r"added.*cart.*recently",
    r"watching this", r"others.*looking", r"purchased a", r"\d+ people",
    r"customers? are", r"booked in the last", r"sold in the last",
]
PRICE_DRIP = [
    r"processing fee", r"service charge", r"convenience fee", r"booking fee",
    r"taxes.*not included", r"resort fee", r"cleaning fee", r"handling cost",
    r"admin surcharge", r"handling fee", r"added at checkout", r"surcharge",
    r"not included in", r"extra fee",
]
DISCOUNT = [
    r"\d+% off", r"save ₹", r"save \$", r"was.*now", r"you save",
    r"coupon", r"discount", r"deal", r"offer",
]
NEG_OPT = [
    r"pre-ticked", r"opted in", r"already selected", r"auto-renew",
    r"pre-selected", r"checked by default", r"pre-checked", r"automatically renew",
    r"auto-debit", r"uncheck", r"leave.*box.*unchecked",
]

# The exact, ordered list of engineered columns the model consumes.
# Keep this list and extract_features() in lock-step.
NUM_COLS = [
    "urgency_kw_count", "scarcity_kw_count", "shame_phrase_flag", "cancel_diff_score",
    "social_proof_flag", "price_drip_flag", "discount_claim_flag", "neg_option_flag",
    "exclamation_count", "question_count", "number_present", "time_reference_flag",
]


def clean_and_lemmatize(text):
    """Lowercase, strip non-alpha (keep ! and ?), tokenize, lemmatize.

    The result feeds the TF-IDF vectorizer. Punctuation is kept so that the
    structural feature extractor can still count it from the *raw* text.
    """
    text_clean = str(text).lower()
    text_clean = re.sub(r"[^a-zA-Z\s!?]", "", text_clean)
    tokens = word_tokenize(text_clean)
    lemmatized = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)


def extract_features(text):
    """Turn a raw UI string into the 12-dimensional interpretable feature dict.

    Returns a plain dict keyed exactly by NUM_COLS so callers can build a
    one-row DataFrame for prediction or stack many rows for training.
    """
    text = str(text)
    text_lower = text.lower()

    return {
        # --- keyword / lexical signals --------------------------------------
        "urgency_kw_count":    sum(bool(re.search(p, text_lower)) for p in URGENCY_KW),
        "scarcity_kw_count":   sum(bool(re.search(p, text_lower)) for p in SCARCITY_KW),
        "shame_phrase_flag":   int(any(re.search(p, text_lower) for p in SHAME_PHRASES)),
        "cancel_diff_score":   sum(bool(re.search(p, text_lower)) for p in CANCEL_DIFF),
        "social_proof_flag":   int(any(re.search(p, text_lower) for p in SOCIAL_PROOF)),
        "price_drip_flag":     int(any(re.search(p, text_lower) for p in PRICE_DRIP)),
        "discount_claim_flag": int(any(re.search(p, text_lower) for p in DISCOUNT)),
        "neg_option_flag":     int(any(re.search(p, text_lower) for p in NEG_OPT)),
        # --- structural signals --------------------------------------------
        "exclamation_count":   text.count("!"),
        "question_count":      text.count("?"),
        "number_present":      int(bool(re.search(r"\d+", text))),
        "time_reference_flag": int(bool(re.search(r"hour|minute|day|tonight|today|soon|week|month|year", text_lower))),
    }
