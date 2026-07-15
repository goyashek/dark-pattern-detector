"""
Dark Pattern Detector — DistilBERT showcase (Hugging Face Space, Route 2).

Companion to the classical Streamlit app. That app is the fast, interpretable default
(character TF-IDF + 12 engineered features -> calibrated LinearSVC, with signal badges). This Space
serves the fine-tuned DistilBERT model — higher accuracy, and the only model that holds up
on real out-of-distribution text, but a black box: it reads whole-phrase meaning instead of
counting keywords.

So the interpretability surface here is different *on purpose*. Instead of faking keyword
badges the transformer doesn't use, we show its full softmax distribution across all 14
classes — what it considered, and how sure it was. That is the honest "why" for a transformer.

Weights load from a separate HF Model repo (set MODEL_ID), so this Space stays light.
"""

import os
import json

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model repo holding the fine-tuned weights. Override via the MODEL_ID Space secret/variable
# if your HF username/repo differs. A local directory also works (handy for local testing).
MODEL_ID = os.environ.get("MODEL_ID", "goyashek/distilbert-darkpattern")

# 14 classes in label-encoder order (alphabetical) — the index order the model's logits use.
# Kept in sync with label_map.json saved alongside the weights; we still try to load that
# file from the repo first, and fall back to this list.
CLASSES = [
    "Bait and Switch", "Basket Sneaking", "Confirm Shaming", "Disguised Advertisement",
    "Drip Pricing", "False Urgency", "Forced Action", "Interface Interference",
    "Nagging", "Not a Dark Pattern", "Rogue Malware", "SaaS Billing",
    "Subscription Trap", "Trick Question",
]
BENIGN = "Not a Dark Pattern"
MAX_LEN = 64

# Precision lever (mirrors the classical app's behaviour): if the top prediction is a
# dark pattern but the model isn't confident enough, fall back to "benign" rather than
# cry wolf. Tune freely — the full distribution is always shown regardless.
CONF_THRESHOLD = 0.50

CCPA_CLAUSES = {
    "False Urgency": "Clause 1 — Falsely implying scarcity/urgency/popular demand to rush a purchase.",
    "Basket Sneaking": "Clause 2 — Adding items or charges to the cart without explicit consent.",
    "Confirm Shaming": "Clause 3 — Using guilt or shame to steer the user's choice.",
    "Forced Action": "Clause 4 — Forcing actions unrelated to the user's goal to proceed.",
    "Subscription Trap": "Clause 5 — Blocking easy cancellation or creating auto-debit loops.",
    "Interface Interference": "Clause 6 — Visual tricks that hide or de-emphasise key options.",
    "Bait and Switch": "Clause 7 — Advertising one thing but delivering another.",
    "Drip Pricing": "Clause 8 — Revealing mandatory fees only at the final checkout step.",
    "Disguised Advertisement": "Clause 9 — Masking ads as organic reviews or content.",
    "Nagging": "Clause 10 — Repeatedly interrupting the user with prompts.",
    "Trick Question": "Clause 11 — Confusing double-negatives or checkboxes to extract consent.",
    "SaaS Billing": "Clause 12 — Silent trial-to-paid billing conversions.",
    "Rogue Malware": "Clause 13 — Fake security/virus alerts to force downloads.",
    "Not a Dark Pattern": "Safe under CCPA Guidelines 2023 — benign UI text.",
}

# A few representative examples per category for one-click testing (gr.Examples).
EXAMPLES = [
    "Hurry! Only 2 items left at this price! Save 20% off!",
    "A premium shipping protection fee has been added to your cart.",
    "No thanks, I prefer to pay full price and remain unprotected.",
    "You must sign up for our newsletter to complete registration.",
    "To cancel your subscription, please call us during business hours.",
    "Get a leather jacket for ₹99! (At checkout, only a plastic cover is included.)",
    "Convenience fee of ₹52 added at the final payment step.",
    "Sponsored listing",
    "Enable push notifications? (Prompted on every page refresh.)",
    "Uncheck this box if you do not want us to not sell your data.",
    "Free 7-day trial. Your card is then charged ₹499/month automatically.",
    "Warning! Your device is infected. Tap to clean it now.",
    "Your order has been placed. Thank you for shopping with us!",
]


def _load():
    """Load tokenizer + model, and the label list (prefer the repo's label_map.json)."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    mdl.eval()

    classes, max_len = CLASSES, MAX_LEN
    # Best effort: use the label map shipped with the weights so this never drifts.
    try:
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(MODEL_ID, "label_map.json")) as f:
            meta = json.load(f)
        classes = meta.get("classes", CLASSES)
        max_len = meta.get("max_length", MAX_LEN)
    except Exception:
        local = os.path.join(MODEL_ID, "label_map.json")
        if os.path.exists(local):
            with open(local) as f:
                meta = json.load(f)
            classes, max_len = meta.get("classes", CLASSES), meta.get("max_length", MAX_LEN)
    return tok, mdl, classes, max_len


TOKENIZER, MODEL, LABELS, MAX_LEN = _load()


def analyze(text):
    """Return (label distribution for gr.Label, verdict markdown)."""
    text = (text or "").strip()
    if not text:
        return {}, "_Enter some UI text above and click Analyze._"

    enc = TOKENIZER(text, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = MODEL(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    dist = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    top = max(range(len(probs)), key=lambda i: probs[i])
    label, conf = LABELS[top], probs[top]

    # weak dark-pattern call -> treat as benign (precision over recall)
    if label != BENIGN and conf < CONF_THRESHOLD:
        label = BENIGN

    if label == BENIGN:
        verdict = (
            "### ✅ CCPA Safe / Benign\n"
            f"**{conf:.0%}** confidence on the top class.\n\n"
            f"{CCPA_CLAUSES[BENIGN]}"
        )
    else:
        verdict = (
            "### 🔴 CCPA Violation Detected\n"
            f"**{label}** — {conf:.0%} confidence\n\n"
            f"{CCPA_CLAUSES.get(label, '')}"
        )
    return dist, verdict


CUSTOM_CSS = """
.gradio-container {max-width: 1040px !important;}
#verdict {min-height: 130px;}
"""

with gr.Blocks(title="Dark Pattern Detector — DistilBERT", theme=gr.themes.Soft(),
               css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# 🔍 Dark Pattern Detector — DistilBERT\n"
        "Fine-tuned transformer that flags deceptive UI copy under **India's CCPA "
        "Dark Pattern Guidelines, 2023** (13 illegal classes + benign). This is the "
        "higher-accuracy companion to the classical Streamlit app — it reads whole-phrase "
        "meaning rather than counting keywords."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="Suspicious UI text",
                placeholder="Paste a button label, warning, urgency banner, fee line…",
                lines=5,
            )
            btn = gr.Button("🚀 Analyze", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp, label="Or try a sample")
        with gr.Column(scale=1):
            verdict = gr.Markdown(elem_id="verdict")
            # gr.Label IS the transformer's interpretability surface: the full confidence
            # spread across all 14 classes, not a fabricated keyword breakdown.
            dist = gr.Label(num_top_classes=5, label="Confidence across classes")

    btn.click(analyze, inputs=inp, outputs=[dist, verdict])
    inp.submit(analyze, inputs=inp, outputs=[dist, verdict])

    with gr.Accordion("Why no keyword badges here? (interpretability note)", open=False):
        gr.Markdown(
            "The **classical** model scores 22 hand-built signals (urgency words, hidden-fee "
            "phrasing, cancellation friction…), so its app can show *which* features fired. "
            "DistilBERT has no such features — it learns phrasing directly. Its honest "
            "interpretability surface is the **probability distribution** above: what it "
            "weighed and how confident it was. When the top dark-pattern class is below "
            f"{int(CONF_THRESHOLD*100)}% it defaults to *benign* to avoid false alarms.\n\n"
            "*Research/educational demo — not legal advice. "
            "Made by [Abhishek Goyal](https://github.com/goyashek).*"
        )

if __name__ == "__main__":
    # HF Spaces runs in a container — bind to 0.0.0.0:7860, not localhost.
    demo.launch(server_name="0.0.0.0", server_port=7860)
