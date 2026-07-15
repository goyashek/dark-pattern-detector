"""Dark Pattern Detector — CCPA 2023 classical text model.

The classifier consumes character TF-IDF plus 12 engineered features through the exact
pipeline saved by training.

    streamlit run app/app.py
"""

import os
import sys
import random
import joblib
import pandas as pd
import streamlit as st

# Make `src` importable whether run from root or app/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import features as F  # noqa: E402

st.set_page_config(
    page_title="Dark Pattern Detector — CCPA 2023",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


@st.cache_resource
def boot():
    F.ensure_nltk()
    models_dir = os.path.join(ROOT, "models")
    multi = joblib.load(os.path.join(models_dir, "best_multi_model.joblib"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    return multi, le


@st.cache_data
def get_samples_by_category():
    path = os.path.join(ROOT, "data", "processed", "ccpa_dataset.tsv")
    samples = {cat: [] for cat in PATTERN_GUIDANCE}
    
    # High quality representative samples for fallback & verification
    fallback = {
        "False Urgency": [
            "Hurry! Only 2 items left at this price! Save 20% off!",
            "Flash sale ends in 5 minutes! Act now before it is gone!",
            "Demand is extremely high. 12 people are looking at this item right now!"
        ],
        "Basket Sneaking": [
            "A $4.99 premium shipping protection fee has been added to your cart.",
            "Adding mandatory insurance cover. Uncheck if you want to decline cover.",
            "Warranty plan added automatically."
        ],
        "Confirm Shaming": [
            "No thanks, I prefer to pay full price and remain unprotected.",
            "No, I hate saving money.",
            "I'd rather risk losing my baggage than pay for premium safety."
        ],
        "Forced Action": [
            "You must sign up for our weekly newsletter to complete your account registration.",
            "Please complete this survey about your preferences to continue to checkout.",
            "Share this on Facebook to download your free report."
        ],
        "Subscription Trap": [
            "To cancel your subscription, please call our customer support department during business hours.",
            "Send a registered letter to our billing address to opt out of the auto-debit scheme.",
            "Cancellation requires scheduling a live call with a retention manager."
        ],
        "Interface Interference": [
            "The 'Cancel subscription' option is hidden in small gray text on a dark background.",
            "Clicking the gray button keeps subscription, green button confirms cancellation.",
            "Visual tricks making it hard to see the unsubscribe option."
        ],
        "Bait and Switch": [
            "Get a premium leather jacket for $10! (At checkout, only a plastic cover is included).",
            "Book a 5-star hotel room for $50. (Upon arrival, shifted to a standard motel room).",
            "Advertise high-end specs, deliver budget refurbished item."
        ],
        "Drip Pricing": [
            "Your ticket is $20. (At final payment, we add $15 service charge and $5 convenience fee).",
            "Base room rate: $80/night. (Mandatory resort fee $25 and cleaning fee $15 added at checkout).",
            "Hidden fees: Total price is revealed only at the very last step."
        ],
        "Disguised Advertisement": [
            "This customer review says: 'I lost 20lbs in 3 days using this product!' (Sponsored post).",
            "Editorial content recommending this service (contains affiliate link with no disclosure).",
            "A review disguised as natural organic customer feedback."
        ],
        "Nagging": [
            "Would you like to sign up for Prime? (Asked every 2 minutes while browsing).",
            "Enable push notifications? (Prompted on every page refresh).",
            "Repeatedly prompting user to upgrade to premium version."
        ],
        "Trick Question": [
            "Uncheck this box if you do not want us to not sell your personal information.",
            "By checking this box, you agree to not opt-out of auto-renewals.",
            "Confusing double negatives: Do not uncheck if you want to opt-out."
        ],
        "SaaS Billing": [
            "Start your free 7-day trial. Your card will be automatically charged $49/month thereafter.",
            "Free trial automatically rolls over to a paid yearly plan unless cancelled today.",
            "Silent subscription renewal: Charged annually without reminder."
        ],
        "Rogue Malware": [
            "Warning! Your computer is infected with 13 viruses. Click here to clean it immediately.",
            "System critical error! Security compromise detected. Download software to fix.",
            "Fake security alert warning you to install malware removal tool."
        ],
        "Not a Dark Pattern": [
            "Add this shirt to your shopping cart.",
            "Click here to read our terms of service.",
            "Your order has been successfully placed. Thank you for shopping with us!",
            "Write a review.",
            "Pillowcases & Shams"
        ],
    }
    
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, sep="\t")
            df = df[["text", "Pattern Category"]].dropna()
            for _, r in df.iterrows():
                cat = r["Pattern Category"]
                text = r["text"].strip()
                if cat in samples and text and text not in samples[cat]:
                    samples[cat].append(text)
        except Exception:
            pass
            
    # Merge and deduplicate
    for cat, items in fallback.items():
        if cat in samples:
            samples[cat] = list(set(samples[cat] + items))
        else:
            samples[cat] = items
            
    return samples


def predict(text, multi, le):
    features = F.extract_features(text)
    model_input = pd.DataFrame([{"text": text, **features}])
    probs = multi.predict_proba(model_input)[0]
    encoded = int(probs.argmax())
    return le.inverse_transform([encoded])[0], float(probs[encoded]), features


def main():
    # Inject Custom Styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Result Cards */
    .signal-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.03) 100%);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(239, 68, 68, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .signal-card:hover {
        border-color: rgba(239, 68, 68, 0.4);
        box-shadow: 0 12px 40px 0 rgba(239, 68, 68, 0.1);
    }
    
    .safe-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(16, 185, 129, 0.03) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .safe-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.1);
    }
    
    /* Custom triggers badges */
    .trigger-badge {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: inline-block;
        margin: 6px;
        transition: all 0.2s ease;
    }
    .trigger-badge:hover {
        background-color: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .trigger-badge.highlight {
        background-color: rgba(239, 68, 68, 0.12);
        color: #fca5a5;
        border-color: rgba(239, 68, 68, 0.25);
    }
    .trigger-badge.highlight:hover {
        background-color: rgba(239, 68, 68, 0.18);
        border-color: rgba(239, 68, 68, 0.4);
    }
    
    .trigger-badge.safe-signal {
        background-color: rgba(16, 185, 129, 0.12);
        color: #6ee7b7;
        border-color: rgba(16, 185, 129, 0.25);
    }
    
    /* Sidebar enhancements */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Main layout headers */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 4px;
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer-link {
        color: #ff7e5f;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.2s;
    }
    .footer-link:hover {
        color: #feb47b;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Header
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 0px 0;">
        <h1 class="hero-title" style="font-size: 2.8rem;">🔍 Dark Pattern Detector</h1>
        <p style="font-size: 1.0rem; color: #94a3b8; max-width: 800px; margin: 0 auto; padding-bottom: 20px;">
            Screening website copy for potential dark-pattern language under the CCPA 2023 Guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        multi, le = boot()
    except Exception as e:
        st.error(f"Could not load models. Run `python -m src.train` first.\n\n{e}")
        return

    # Load samples
    samples = get_samples_by_category()

    # Sidebar Content & Disclaimers
    st.sidebar.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #f1f5f9; font-size: 1.1rem; font-family: 'Outfit', sans-serif;">⚙️ Classifier Info</h3>
        <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.4; margin-bottom: 0;">
            Research classifier mapping UI text onto India's 13 named dark-pattern categories plus a no-dark-pattern class.
            Model: character TF-IDF + 12 engineered features + SMOTE + calibrated LinearSVC.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="background-color: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.15); border-radius: 12px; padding: 16px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #fca5a5; font-size: 1.0rem; font-family: 'Outfit', sans-serif;">⚠️ Legal Disclaimer</h3>
        <p style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.4; margin-bottom: 0;">
            This is a <strong>student project</strong>. I mapped the dataset labels to the CCPA dark-pattern categories based on my own reading of the guidelines. The mapping is not official or approved by the CCPA, and the results should not be used as legal or compliance advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0; border-top: 1px solid rgba(255,255,255,0.05);">
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">
            Made by <a href="https://github.com/goyashek" target="_blank" class="footer-link">Abhishek Goyal</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize tabbed interface
    tab_detector, tab_about = st.tabs(["🔍 Detector Dashboard", "📚 About the Project"])

    with tab_detector:
        # Sleek, collapsed expander for Demo Samples to declutter page load
        with st.expander("💡 Click here to test using a pre-loaded Demo Sample instead", expanded=False):
            col_sample_sel, col_sample_btn = st.columns([3.2, 0.8], gap="medium")
            with col_sample_sel:
                selected_cat = st.selectbox(
                    "Select Category to Sample:",
                    list(PATTERN_GUIDANCE),
                    format_func=lambda label: DISPLAY_LABELS.get(label, label),
                    key="sample_sel",
                    label_visibility="collapsed",
                )
            with col_sample_btn:
                if st.button("🎲 Load Sample Text", use_container_width=True):
                    cat_samples = samples.get(selected_cat, [])
                    if cat_samples:
                        st.session_state["ui_text_input"] = random.choice(cat_samples)
                        st.session_state["analyzed"] = False
                        st.rerun()

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Main columns
        col_input, col_results = st.columns([1.1, 0.9], gap="large")

        with col_input:
            st.markdown('<h3 style="font-size: 1.25rem; font-family: \'Outfit\', sans-serif; margin-bottom: 12px;">📝 Suspicious UI Text Input</h3>', unsafe_allow_html=True)
            
            # Initialize text area input in session state if not present
            if "ui_text_input" not in st.session_state:
                st.session_state["ui_text_input"] = "Hurry! Only 2 items left at this price! Save 20% off!"
                
            text = st.text_area(
                "suspicious_ui_copy_field",
                help="Paste any suspicious UI copy, warnings, urgency labels, or buttons found on websites.",
                height=140, 
                key="ui_text_input",
                label_visibility="collapsed"
            )
            
            analyze_btn = st.button("🚀 Analyze Copy", type="primary", use_container_width=True)
            if analyze_btn:
                st.session_state["analyzed"] = True

        with col_results:
            st.markdown('<h3 style="font-size: 1.25rem; font-family: \'Outfit\', sans-serif; margin-bottom: 12px;">📊 Analysis Results</h3>', unsafe_allow_html=True)
            
            # Only predict if the user has clicked "Analyze Copy"
            if st.session_state.get("analyzed", False) and text:
                label, conf, feats = predict(text, multi, le)
                description, context = PATTERN_GUIDANCE[label]
                display_label = DISPLAY_LABELS.get(label, label)
                
                # Display result card
                if label != "Not a Dark Pattern":
                    st.markdown(f"""
                    <div class="signal-card">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #f87171; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px;">
                            🔴 POTENTIAL TEXTUAL SIGNAL
                        </div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #fef2f2; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">
                            {display_label}
                        </div>
                        <div style="font-size: 0.95rem; color: #fca5a5; margin-bottom: 16px; line-height: 1.4;">
                            {description}
                        </div>
                        <div style="display: inline-block; background-color: rgba(239, 68, 68, 0.15); color: #fca5a5; padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; border: 1px solid rgba(239, 68, 68, 0.25);">
                            Top-class probability: {conf:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #34d399; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px;">
                            ✅ NO TEXTUAL SIGNAL FOUND
                        </div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #ecfdf5; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">
                            No textual signal
                        </div>
                        <div style="font-size: 0.95rem; color: #a7f3d0; margin-bottom: 16px; line-height: 1.4;">
                            {description}
                        </div>
                        <div style="display: inline-block; background-color: rgba(16, 185, 129, 0.15); color: #a7f3d0; padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; border: 1px solid rgba(16, 185, 129, 0.25);">
                            Top-class probability: {conf:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.info(f"Context needed: {context} This screening result does not establish compliance or a violation.")
                
                # Extract and show only active triggers
                st.markdown('<p style="font-weight: 600; font-size: 1.05rem; margin-bottom: 6px; font-family: \'Outfit\', sans-serif;">🔍 Lexical Signals</p>', unsafe_allow_html=True)
                triggers = []
                if feats.get("urgency_kw_count", 0) > 0:
                    triggers.append(("urgency", f"🚨 Urgency keywords ({feats['urgency_kw_count']})"))
                if feats.get("scarcity_kw_count", 0) > 0:
                    triggers.append(("scarcity", f"📦 Scarcity signals ({feats['scarcity_kw_count']})"))
                if feats.get("shame_phrase_flag", 0) > 0:
                    triggers.append(("shame", "💬 Confirm-shaming phrasing"))
                if feats.get("cancel_diff_score", 0) > 0:
                    triggers.append(("cancel", f"🚧 Cancellation obstruction ({feats['cancel_diff_score']})"))
                if feats.get("social_proof_flag", 0) > 0:
                    triggers.append(("social", "👥 Social-proof phrasing"))
                if feats.get("price_drip_flag", 0) > 0:
                    triggers.append(("price", "💰 Hidden-fee / drip-pricing"))
                if feats.get("discount_claim_flag", 0) > 0:
                    triggers.append(("discount", "🏷️ Discount/offer claim"))
                if feats.get("neg_option_flag", 0) > 0:
                    triggers.append(("neg", "☑️ Negative-option/pre-checked"))
                if feats.get("exclamation_count", 0) > 0:
                    triggers.append(("structure", f"❗ Exclamations ({feats['exclamation_count']})"))
                if feats.get("time_reference_flag", 0) > 0:
                    triggers.append(("time", "⏰ Time reference"))
                    
                if triggers:
                    badge_html = '<div style="margin: -6px 0 16px 0;">'
                    for tag, text_val in triggers:
                        is_risk_signal = tag in ["urgency", "scarcity", "shame", "cancel", "social", "price", "neg"]
                        badge_class = "trigger-badge highlight" if is_risk_signal else "trigger-badge"
                        badge_html += f'<span class="{badge_class}">{text_val}</span>'
                    badge_html += '</div>'
                    st.markdown(badge_html, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="margin-bottom: 16px;"><span class="trigger-badge safe-signal">🌿 No listed lexical signals fired</span></div>', unsafe_allow_html=True)
                    
                with st.expander("🔬 View raw feature values"):
                    st.json({k: feats[k] for k in F.NUM_COLS})
            else:
                # Beautiful dashed placeholder card for a clean initial state
                st.markdown("""
                <div style="border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 16px; padding: 42px 20px; text-align: center; color: #64748b; margin-top: 5px;">
                    <div style="font-size: 2.8rem; margin-bottom: 14px;">🔍</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px;">Awaiting Audit Input</div>
                    <p style="font-size: 0.85rem; color: #64748b; margin: 0; line-height: 1.45; max-width: 280px; margin: 0 auto;">
                        Enter website UI copy on the left and click <strong>Analyze Copy</strong> to view the risk-screening result.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    with tab_about:
        st.markdown("""
        ### About the CCPA Dark Pattern Detector
        
        This project is a research risk screener organized around the categories in **India's Consumer Protection (Prevention and Regulation of Dark Patterns) Guidelines, 2023**.
        
        #### Key Objectives
        - **Promote User Autonomy**: Identify interface copy designed to trick, coerce, or manipulate consumers.
        - **Support Human Review**: Surface possible textual signals without making a legal determination.
        - **Machine Learning Powered Audit**: Uses a compact classical text classifier to analyze snippets.
        
        #### Technical Architecture
        1. **Character TF-IDF**: Keeps short phrases, punctuation, prices, and spelling variants.
        2. **12 engineered features**: Adds focused lexical and structural signals.
        3. **SMOTE + LinearSVC**: Balances the training fold and classifies into 14 categories.
        4. **Grouped calibration**: Produces probabilities without mixing page/template groups.
        5. **Direct output**: Shows the calibrated winning-class probability without a fallback threshold.
        
        #### The 13 CCPA Dark-Pattern Categories
        """)
        
        for k, (description, context) in PATTERN_GUIDANCE.items():
            if k != "Not a Dark Pattern":
                st.markdown(f"- **{DISPLAY_LABELS.get(k, k)}**: {description} *Context needed: {context}*")
                
        st.markdown("""
        ---
        
        #### 👨‍💻 Project Information & Credits
        - **Created By**: [Abhishek Goyal](https://github.com/goyashek)
        - **Repository Link**: [GitHub Profile](https://github.com/goyashek)
        
        #### ⚠️ Disclaimer
        *This is a student project. I mapped the dataset labels to the CCPA dark-pattern categories based on my own reading of the guidelines. The mapping is not official or approved by the CCPA, and the results should not be used as legal or compliance advice.*
        """)


if __name__ == "__main__":
    main()
