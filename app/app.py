"""Streamlit app for the classical dark-pattern text model."""

from pathlib import Path
import random
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402

st.set_page_config(page_title="Dark Pattern Detector", page_icon="🔍")

PATTERN_GUIDANCE = {
    "False Urgency": ("Language resembling urgency or scarcity.", "Verify the timer, inventory, demand, and offer expiry."),
    "Basket Sneaking": ("Language resembling an added item or charge.", "Compare the cart before and after the user's explicit choices."),
    "Confirm Shaming": ("Language that may shame or guilt a user.", "Review the surrounding choices and whether refusal is neutral."),
    "Forced Action": ("Language suggesting an extra action may be required.", "Check whether the user's task is blocked by an unrelated action."),
    "Subscription Trap": ("Language resembling enrollment or cancellation friction.", "Review the complete signup, renewal, and cancellation flow."),
    "Interface Interference": ("Language associated with possible interface interference.", "Inspect layout, defaults, contrast, prominence, and nearby controls."),
    "Bait and Switch": ("Language resembling a changed offer or outcome.", "Compare the original offer, selected action, and actual result."),
    "Drip Pricing": ("Language resembling late price disclosure.", "Compare every price shown from listing through final payment."),
    "Disguised Advertisement": ("Language resembling promotional content.", "Check sponsorship, placement, and disclosure context."),
    "Nagging": ("Language associated with repeated prompting.", "Observe how often and when the prompt reappears."),
    "Trick Question": ("Language resembling confusing consent wording.", "Review checkbox defaults, nearby wording, and the effect of each choice."),
    "SaaS Billing": ("Language associated with trial or recurring billing.", "Verify renewal terms, consent, reminders, and cancellation steps."),
    "Rogue Malware": ("Language resembling an alarming security prompt.", "Verify the source, device state, requested action, and download target."),
    "Not a Dark Pattern": ("The model found no category-level textual signal in this snippet.", "Review the surrounding interface and full user flow before drawing a conclusion."),
}
DISPLAY_LABELS = {"Trick Question": "Trick Wording (model label: Trick Question)"}
SIGNAL_NAMES = {
    "urgency_kw_count": "urgency words",
    "scarcity_kw_count": "scarcity words",
    "shame_phrase_flag": "confirm-shaming wording",
    "cancel_diff_score": "cancellation friction",
    "social_proof_flag": "social-proof wording",
    "price_drip_flag": "late-fee wording",
    "discount_claim_flag": "discount claim",
    "neg_option_flag": "negative-option wording",
    "exclamation_count": "exclamation marks",
    "question_count": "question marks",
    "number_present": "number present",
    "time_reference_flag": "time reference",
}


@st.cache_resource
def boot():
    return (
        joblib.load(ROOT / "models/best_multi_model.joblib"),
        joblib.load(ROOT / "models/label_encoder.joblib"),
    )


@st.cache_data
def get_samples_by_category():
    df = pd.read_csv(ROOT / "data/processed/ccpa_dataset.tsv", sep="\t")
    df = df.dropna(subset=["text", "Pattern Category"])
    return {
        category: rows["text"].drop_duplicates().tolist()
        for category, rows in df.groupby("Pattern Category")
    }


def predict(text, model, encoder):
    values = F.extract_features(text)
    probabilities = model.predict_proba(pd.DataFrame([{"text": text, **values}]))[0]
    winner = int(probabilities.argmax())
    return encoder.inverse_transform([winner])[0], float(probabilities[winner]), values


def main():
    st.title("🔍 Dark Pattern Detector")
    st.caption("Checks short UI text for possible dark-pattern wording using a student-built classifier.")

    st.sidebar.subheader("Model")
    st.sidebar.write("Character TF-IDF, 12 text features, SMOTE and a calibrated LinearSVC.")
    st.sidebar.warning(
        "This is a student project. The label mapping is based on my reading of the CCPA "
        "guidelines and is not official or legal advice."
    )

    try:
        model, encoder = boot()
        samples = get_samples_by_category()
    except Exception as error:
        st.error(f"Could not load the saved project files: {error}")
        return

    st.session_state.setdefault(
        "ui_text_input", "Hurry! Only 2 items left at this price! Save 20% off!"
    )

    with st.expander("Try a saved example"):
        category = st.selectbox(
            "Category",
            list(PATTERN_GUIDANCE),
            format_func=lambda label: DISPLAY_LABELS.get(label, label),
        )
        if st.button("Load example") and samples.get(category):
            st.session_state["ui_text_input"] = random.choice(samples[category])
            st.rerun()

    text = st.text_area("UI text", key="ui_text_input", height=140)
    if not st.button("Analyze", type="primary"):
        return
    if not text.strip():
        st.warning("Enter some UI text first.")
        return

    label, score, values = predict(text, model, encoder)
    description, context = PATTERN_GUIDANCE[label]
    shown_label = DISPLAY_LABELS.get(label, label)

    if label == "Not a Dark Pattern":
        st.success(f"No textual signal found ({score:.1%} top-class probability)")
    else:
        st.error(f"Potential textual signal: {shown_label} ({score:.1%} top-class probability)")
    st.write(description)
    st.info(f"Context needed: {context} This result does not establish compliance or a violation.")

    active = [
        f"{SIGNAL_NAMES[name]} ({value})"
        for name, value in values.items()
        if value
    ]
    st.write("Text signals:", ", ".join(active) if active else "none of the 12 listed signals")

    with st.expander("Raw feature values"):
        st.json({name: values[name] for name in F.NUM_COLS})

    with st.expander("How the project works"):
        st.write(
            "The model combines character n-grams with 12 hand-built text features. "
            "It predicts one of 13 CCPA categories or the no-dark-pattern class. "
            "A full interface review is still needed because this app only reads text."
        )


if __name__ == "__main__":
    main()
