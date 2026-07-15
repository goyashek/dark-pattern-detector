"""
Dark Pattern Detector — DistilBERT showcase (Hugging Face Space, Route 2).

Companion to the classical Streamlit app. That app is the fast, interpretable default
(character TF-IDF + 12 engineered features -> calibrated LinearSVC, with signal badges). This Space
serves the fine-tuned DistilBERT model, a black box that reads whole-phrase meaning instead
of counting keywords.

So the interpretability surface here is different *on purpose*. Instead of faking keyword
badges the transformer doesn't use, we show its full softmax distribution across all 14
classes. These softmax scores show what it considered, but are not calibrated confidence.

Weights load from a separate HF Model repo (set MODEL_ID), so this Space stays light.
"""

import os
import json

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from .presentation import ABSTAIN_THRESHOLD, BENIGN, result_status
except ImportError:  # HF Spaces uploads this directory as the repository root.
    from presentation import ABSTAIN_THRESHOLD, BENIGN, result_status

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
MAX_LEN = 64

# Provisional UI band only; it has not been selected as a legal or operational threshold.
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
    BENIGN: ("The model found no category-level textual signal in this snippet.", "Review the surrounding interface and full user flow before drawing a conclusion."),
}
DISPLAY_LABELS = {"Trick Question": "Trick Wording (model label: Trick Question)"}

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
    dist = {DISPLAY_LABELS.get(LABELS[i], LABELS[i]): float(probs[i]) for i in range(len(LABELS))}

    top = max(range(len(probs)), key=lambda i: probs[i])
    label, conf = LABELS[top], probs[top]

    status = result_status(label, conf)
    description, context = PATTERN_GUIDANCE[label]
    display_label = DISPLAY_LABELS.get(label, label)

    if status == "inconclusive":
        verdict = (
            "### ⚪ Inconclusive from text alone\n"
            f"Leading model category: **{display_label}** ({conf:.0%} softmax score).\n\n"
            f"This is below the provisional {ABSTAIN_THRESHOLD:.0%} display threshold; "
            "it is not relabeled as benign.\n\n"
            f"**Context needed:** {context}"
        )
    elif status == "no_signal":
        verdict = (
            "### ✅ No textual signal found\n"
            f"**{conf:.0%}** softmax score for the top class.\n\n"
            f"{description}\n\n**Context needed:** {context}"
        )
    else:
        verdict = (
            "### 🔴 Potential textual signal\n"
            f"**{display_label}** — {conf:.0%} softmax score\n\n"
            f"{description}\n\n**Context needed:** {context}"
        )
    verdict += "\n\n*Screening result only; human review is required for any compliance conclusion.*"
    return dist, verdict


CUSTOM_CSS = """
.gradio-container {max-width: 1040px !important;}
#verdict {min-height: 130px;}
"""

with gr.Blocks(title="Dark Pattern Detector — DistilBERT", theme=gr.themes.Soft(),
               css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# 🔍 Dark Pattern Detector — DistilBERT\n"
        "Research risk screener trained against the 13 categories named in **India's CCPA "
        "Dark Pattern Guidelines, 2023**, plus a no-dark-pattern class. This is the "
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
            # gr.Label is the transformer's interpretability surface: the full softmax
            # spread across all 14 classes, not a fabricated keyword breakdown.
            dist = gr.Label(num_top_classes=5, label="Model scores across classes")

    btn.click(analyze, inputs=inp, outputs=[dist, verdict])
    inp.submit(analyze, inputs=inp, outputs=[dist, verdict])

    with gr.Accordion("Why no keyword badges here? (interpretability note)", open=False):
        gr.Markdown(
            "The **classical** model scores 12 hand-built signals (urgency words, hidden-fee "
            "phrasing, cancellation friction…), so its app can show *which* features fired. "
            "DistilBERT has no such features — it learns phrasing directly. Its honest "
            "interpretability surface is the **softmax score distribution** above: what it "
            "weighed, not calibrated confidence. Any top score below the provisional "
            f"{int(ABSTAIN_THRESHOLD*100)}% display threshold is shown as *inconclusive*, "
            "never converted to benign.\n\n"
            "*This is a student project. I mapped the dataset labels to the CCPA dark-pattern "
            "categories based on my own reading of the guidelines. The mapping is not official "
            "or approved by the CCPA, and the results should not be used as legal or compliance "
            "advice. Made by [Abhishek Goyal](https://github.com/goyashek).*"
        )

if __name__ == "__main__":
    # HF Spaces runs in a container — bind to 0.0.0.0:7860, not localhost.
    demo.launch(server_name="0.0.0.0", server_port=7860)
