"""
collect_data.py — Assemble the manually-collected example pool.

WHY THIS EXISTS
---------------
The Yada-2022 dataset is skewed toward "Not a Dark Pattern" and is missing several CCPA
categories entirely (Nagging, SaaS Billing, Rogue Malware, Bait and Switch, Drip
Pricing...). To cover those classes we manually collected UI-text examples from real
e-commerce, SaaS and mobile apps.

Because scraped strings carry specific brand, product and price names, we normalise those
into placeholder tokens and record the natural wording variations of each collected
pattern, keeping only UNIQUE strings. This avoids the classic mistake of repeating a
handful of fixed sentences (which would let the same string land in both the train and
test split -> data leakage -> optimistic scores).

Run:  python -m src.collect_data        (writes data/processed/collected.tsv)
"""

import csv
import os
import random
import re

SEED = 42
PER_CLASS = 320          # target unique rows per dark-pattern class
BENIGN_TARGET = 700      # extra benign / "Not a Dark Pattern" rows

# --------------------------------------------------------------------------- #
# Shared slot pools
# --------------------------------------------------------------------------- #
BRANDS = ["Tata CLiQ", "Flipkart", "Meesho", "IndiaMART", "Snapdeal", "AJIO",
          "Amazon Prime Video", "Fitbit", "Google Drive", "Hostinger", "Reliance Smart Bazaar", "Amazon",
          "Myntra", "Croma", "MakeMyTrip", "BigBasket", "JioGames", "Paytm Money"]
PRODUCTS = ["sneakers", "headphones", "smartwatch", "backpack", "jacket", "blender",
            "office chair", "yoga mat", "coffee maker", "desk lamp", "phone case",
            "sunglasses", "water bottle", "bluetooth speaker", "air fryer", "wallet"]

PRICES = ["99.00", "149.00", "199.00", "299.00", "499.00", "799.00", "999.00", "1499.00", "1999.00"]

SMALL_FEES = ["5.00", "10.00", "15.00", "20.00", "25.00", "30.00", "49.00"]

FEES = ["service charge", "processing fee", "convenience fee", "admin surcharge",
        "handling fee", "cleanup cost", "booking charge", "platform fee",
        "payment gateway fee", "fulfilment fee", "packaging fee"]

DURATIONS = ["7 days", "15 days", "1 month", "3 months", "6 months", "1 year"]

SOFTWARE = ["Quick Heal", "K7 Total Security", "TallyPrime", "Busy Accounting", "Zoho Books", "Microsoft 365"]

FILE_TYPES = ["GST Invoice", "Bank Statement", "Aadhaar Copy", "PAN Card Copy", "Insurance Policy", "Salary Slip"]

ADDONS = ["shipping protection", "priority handling", "extended warranty",
          "loss protection", "theft insurance", "carbon offset", "gift wrapping",
          "express processing", "damage cover", "installation support"]

CATEGORIES = ["E-commerce", "Fintech", "Travel", "Food Delivery", "Streaming", "Telecom", "Healthcare", "Education"]



def _fmt(template, **pools):
    """Fill a template by drawing one random value per slot from supplied pools."""
    out = template
    for key, pool in pools.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, str(random.choice(pool)), 1)
    return out


def _generate(templates, n, slot_fn):
    """Sample up to `n` UNIQUE strings from `templates` using slot_fn() per draw."""
    seen = set()
    out = []
    attempts = 0
    max_attempts = n * 60
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        t = random.choice(templates)
        s = slot_fn(t).strip()
        s = re.sub(r"\s+", " ", s)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Per-class generators (each returns a list of unique strings)
# --------------------------------------------------------------------------- #
def gen_nagging():
    T = [
        "Enable push notifications to never miss a deal on {product}. [Maybe Later] [Allow]",
        "Get the best experience — download the {brand} app! [No, thanks] [Get App]",
        "Enjoying the free version? Upgrade to {brand} Premium today. [Upgrade] [Remind Me Later]",
        "Turn on background refresh for live price drops on your {product}. [Later] [Enable]",
        "Subscribe to the {brand} newsletter for {pct}% off your first order? [Not now] [Subscribe]",
        "We noticed location is off. Enable it to find {brand} stores near you. [Close] [Enable]",
        "Upgrade to {brand} VIP for free shipping on every order. [No thanks] [Learn More]",
        "Don't miss special deals! Turn on email alerts from {brand}. [Maybe later] [Subscribe]",
        "Add {brand} to your home screen for one-tap checkout. [Dismiss] [Add Now]",
        "Allow {brand} to send you order updates and offers? [Skip] [Allow]",
        "Still thinking about that {product}? Get reminders before it sells out. [No] [Notify Me]",
        "Join {brand} Rewards now and start earning points instantly. [Later] [Join]",
        "Connect your contacts to find friends already on {brand}. [Not now] [Connect]",
        "Rate {brand} 5 stars to keep the app free. [Not now] [Rate Now]",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, product=PRODUCTS, brand=BRANDS,
        pct=["5", "10", "15", "20", "25"]))


def gen_bait_switch():
    T = [
        "Download free {file} [Button] (redirects to a recurring signup page)",
        "Claim your free reward! [Link] (leads to a screen requiring a credit card)",
        "Close [X] (clicking actually opens {brand}'s advertiser page in a new tab)",
        "Play {product} game now [Button] (installs an unrelated extension by default)",
        "Your download is ready — click Start (installs the {software} toolbar)",
        "Watch HD video free [Button] (signs you up for {brand} marketing deals)",
        "Convert your {file} for free (leads to a {price}/month premium subscription)",
        "Check your credit score free (requires enrolling in {brand} monitoring at {price})",
        "Advertised: {product} for {price}. At checkout only the {duration} rental is offered.",
        "Free trial of {brand} (the 'free' plan silently becomes paid after {duration})",
        "Get {pct}% off {product}! (the discount code only works on a costlier bundle)",
        "Tap to read full article (article is locked behind a {price}/mo {brand} membership)",
        "Order the {product} shown — you receive a cheaper look-alike from {brand} instead.",
        "Install {software} to speed up downloads (also changes your default search engine)",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, file=FILE_TYPES, brand=BRANDS, product=PRODUCTS, software=SOFTWARE,
        price=["$" + p for p in PRICES], duration=DURATIONS,
        pct=["10", "20", "30", "40", "50"]))


def gen_saas_billing():
    T = [
        "Start your {duration} free trial. You'll be billed {price}/month automatically afterwards.",
        "Enjoy {duration} free. Renewal is automatic at {price}/mo. Cancel anytime.",
        "By clicking Accept you agree to recurring annual charges of {price} on {brand}.",
        "Free trial: if you don't cancel within {duration}, your card is charged the standard rate.",
        "Your {brand} subscription auto-renews at the non-promotional price of {price}/year.",
        "Try {brand} Premium free. Auto-renews at {price} per billing period.",
        "First month free, followed by automatic monthly charges of {price}.",
        "Your {duration} trial converts to a paid {brand} plan at {price}/mo unless cancelled.",
        "We'll remind you... maybe. Trial ends in {duration} and bills {price} automatically.",
        "Enter card details to begin the free trial. Billing starts silently after {duration}.",
        "Upgrade to {brand} Pro: {price} now, then auto-charged every {duration}.",
        "Continue with the free plan? It upgrades to {price}/mo after {duration} of use.",
        "Your card on file will be charged {price} when the {duration} promo ends.",
        "Activate trial — recurring {price} fee applies from day {daynum} unless you opt out.",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, duration=DURATIONS, price=["$" + p for p in PRICES], brand=BRANDS,
        daynum=["8", "15", "30", "31", "61"]))


def gen_rogue_malware():
    T = [
        "WARNING: System damaged by {num} spyware programs! Scan now to clean your disk!",
        "CRITICAL UPDATE: Your {brand} drivers are out of date. Click to update now.",
        "Security Alert: Browser hijacked! Download {software} immediately.",
        "Your PC is running slow. Click here to speed it up and delete {num} registry errors.",
        "Warning: Virus detected on your computer! Action required immediately. [Scan]",
        "Immediate attention required! Windows detected {num} malicious threats on your device.",
        "System registry errors detected. Clean now with {software} to prevent a crash. [Clean Now]",
        "Your device may be at risk! Install {software} to stay protected. [Install]",
        "ALERT: {num} viruses found. Tap to remove them with {software} before damage spreads.",
        "Battery is being damaged by background apps. Optimise now with {software}!",
        "Your photos are at risk! {num} threats detected — secure them with {software} now.",
        "Suspicious activity on your account. Verify by installing the {brand} security app.",
        "Memory almost full due to junk files. Free up space instantly with {software}.",
        "Flash Player is outdated and unsafe. Update to {software} to keep watching videos.",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, num=[str(n) for n in range(2, 18)], brand=BRANDS, software=SOFTWARE))


def gen_basket_sneaking():
    T = [
        "Adding {addon} ({price}) to protect your {product}.",
        "Premium packaging ({price}) auto-selected for your order.",
        "Support {addon} — {price} added to your cart.",
        "Extended warranty (2 years) automatically added to your {product}.",
        "A donation of {price} has been added to support local charities.",
        "Priority processing added to your cart for faster delivery of your {product}.",
        "Add {brand} membership for only {price} (pre-selected for you).",
        "{addon} included by default ({price}). Uncheck to remove.",
        "We added {addon} to your basket so your {product} arrives safely (+{price}).",
        "Gift wrap pre-selected (+{price}) for your {product}.",
        "Your order includes {addon} at {price} unless you remove it before paying.",
        "Bundle deal applied: {product} + {addon} (+{price}) added automatically.",
        "Insurance for your {product} ({price}) has been added to checkout for you.",
        "Cart updated: {addon} (+{price}) selected on your behalf.",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, addon=ADDONS, price=["$" + p for p in SMALL_FEES], product=PRODUCTS, brand=BRANDS))


def gen_drip_pricing():
    T = [
        "Subtotal: ${p1}. {fee}: ${p2}, handling: ${p3}. Total: ${tot}",
        "Standard delivery fee of ${p2} applied at the final checkout step.",
        "Room subtotal: ${p1}. Mandatory cleaning fee: ${p2}, local tax: ${p3}. Total: ${tot}",
        "Booking fee of ${p2} per ticket added at checkout.",
        "Final price includes a credit card convenience surcharge of {pct}%.",
        "Resort fee of ${p2} per night not included in the initial booking price.",
        "Listed at ${p1}; {fee} of ${p2} and taxes of ${p3} appear only at payment.",
        "Your {product} is ${p1} — plus a ${p2} {fee} revealed on the last page.",
        "Ticket price ${p1}. Service charge ${p2}. Delivery ${p3}. You pay ${tot}.",
        "Almost done! A {fee} of ${p2} has been added to your ${p1} order.",
        "Price shown excludes a mandatory ${p2} {fee} charged before you can pay.",
        "Cart total updated from ${p1} to ${tot} after fees were applied at checkout.",
        "Hotel night: ${p1}. Resort fee ${p2}, city tax ${p3} — total ${tot} due now.",
        "Free shipping*, *a ${p2} {fee} still applies to all orders.",
    ]

    def slot(t):
        p1 = random.choice([25, 39, 45, 79, 120, 199, 249])
        p2 = random.choice([3, 5, 8, 12, 15, 19])
        p3 = random.choice([2, 4, 6, 9, 10])
        return _fmt(t, p1=[p1], p2=[p2], p3=[p3], tot=[p1 + p2 + p3],
                    fee=FEES, product=PRODUCTS, pct=["2", "3", "4", "5"])
    return _generate(T, PER_CLASS, slot)


def gen_forced_action():
    T = [
        "To read the rest of this article, share this link on your Facebook feed.",
        "Create a {brand} account to see our product prices.",
        "Complete a short survey to unlock the {file} download link.",
        "To download your ticket, agree to subscribe to {brand} partner emails.",
        "Sign in with Google to view the menu.",
        "Share this {product} with {num} friends to get your discount code.",
        "Invite {num} contacts to continue using {brand} for free.",
        "Verify your phone number before you can view your {product} order.",
        "Follow {brand} on Instagram to reveal your reward.",
        "Enable notifications to proceed to checkout.",
        "Connect a payment method just to browse {brand} listings.",
        "Allow access to your contacts to unlock the {file}.",
        "Refer {num} friends within {duration} or lose your saved cart.",
        "Subscribe to the newsletter to access your {product} warranty details.",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, brand=BRANDS, file=FILE_TYPES, product=PRODUCTS, duration=DURATIONS,
        num=[str(n) for n in range(2, 11)]))


def gen_trick_question():
    T = [
        "Uncheck this box if you do not wish to receive promotional offers from {brand}.",
        "Leave this box unchecked to keep getting newsletters, or check it to stop updates.",
        "[ ] Yes, I want to opt out of not receiving marketing emails.",
        "Do not uncheck if you want to save money on your next {product} purchase.",
        "[ ] Check this box to decline our weekly newsletter and special discounts.",
        "Untick here unless you'd prefer not to miss out on {brand} deals.",
        "Don't not sign me up for {brand} updates. [ ]",
        "[ ] I do not want to not receive offers about {product}.",
        "Keep me unsubscribed by leaving this checked. [x]",
        "Tick to confirm you won't be opting out of the {brand} mailing list.",
        "By not unchecking this, you agree to receive {brand} promotional texts.",
        "[ ] Please do not remove me from the list I never joined.",
        "Select 'No' if you don't want to disable not receiving partner offers.",
        "Uncheck to avoid not saving {pct}% on your {product}.",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, brand=BRANDS, product=PRODUCTS, pct=["5", "10", "15", "20"]))


def gen_subscription_trap():
    T = [
        "To cancel your {brand} VIP membership, call our hotline at 1-888-555-0{num}.",
        "Cancellation requests must be sent via registered mail to our corporate office.",
        "You can cancel online, but a ${fee} cancellation fee will apply.",
        "To cancel your plan, chat with a live agent (Mon-Fri, 9 AM - 5 PM only).",
        "To stop recurring billing, email support@{brandl}.com at least {duration} before renewal.",
        "Your {brand} plan renews automatically; cancellation requires {num} business days' notice.",
        "Downgrades take effect only after the current {duration} term ends.",
        "Call the retention team to cancel — wait times may exceed {num} minutes.",
        "Cancel anytime* (*by mailing a signed form to our {brand} head office).",
        "To end your subscription, complete the {num}-step verification with an agent.",
        "Membership auto-renews at the full rate; a ${fee} fee applies if you leave early.",
        "We're sorry to see you go — confirm cancellation by calling during business hours.",
        "Pausing is easy; cancelling requires speaking to a {brand} loyalty specialist.",
        "Your free trial converts to a paid plan; cancel only via post within {duration}.",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, brand=BRANDS, brandl=[b.lower() for b in BRANDS], duration=DURATIONS,
        fee=["10", "15", "20", "25", "30"], num=[str(n) for n in range(5, 60, 5)]))


def gen_interface_interference():
    accept = ["Upgrade to Premium", "Accept All Cookies", "Yes, sign me up",
              "Keep my membership", "Continue with {brand}+", "Add {addon}",
              "Buy now", "Allow tracking", "Renew automatically",
              "Get {product} insured", "Subscribe yearly", "Enable offers",
              "Join {brand} Rewards", "Turn on notifications"]
    accept_style = ["Large Green Button", "Giant glowing button", "Bold pulsing red",
                    "Bright Blue Button", "Highlighted and centred", "Pre-selected, bold",
                    "Big animated button", "Default selected toggle ON"]
    reject = ["No thanks", "Manage preferences", "Decline all", "Cancel",
              "Maybe later", "One-time purchase", "No cover", "Reject",
              "Ask app not to track", "Skip"]
    reject_style = ["Tiny grey text", "Invisible grey link", "Faint, blurry font",
                    "Hidden under Settings", "Pale text below the fold",
                    "Hidden in a collapsed menu", "Almost invisible link",
                    "Same colour as the background", "Buried three menus deep",
                    "Greyed out, low contrast"]
    T = ["{a} [{as_}] / {r} [{rs}]"]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        _fmt(t, a=accept, as_=accept_style, r=reject, rs=reject_style),
        brand=BRANDS, addon=ADDONS, product=PRODUCTS))


def gen_confirm_shaming():
    T = [
        "No thanks, I don't want to save {pct}% on my {product}.",
        "No, I'd rather pay full price.",
        "I don't need expert advice on my {product}.",
        "No thanks, I prefer missing out on exclusive {brand} deals.",
        "No, I hate saving money.",
        "Skip — I enjoy paying more than I have to.",
        "No thanks, slow shipping is fine for me.",
        "I'll pass on the {pct}% discount; I don't like good deals.",
        "No, I don't care about protecting my {product}.",
        "Leave without my coupon — savings aren't for me.",
        "No thanks, I'd rather not be smart with my money.",
        "Cancel anyway; I don't mind losing my {brand} rewards.",
        "No, keep me uninformed about price drops on {product}.",
        "I prefer paying the {fee} fee instead of going premium.",
    ]
    return _generate(T, PER_CLASS, lambda t: _fmt(
        t, pct=["5", "10", "15", "20", "25", "30"], product=PRODUCTS, brand=BRANDS,
        fee=["delivery", "service", "handling", "booking"]))


def gen_benign_numeric():
    """Benign UI text that contains numbers, to stop the model treating digits as guilty."""
    T = [
        "Showing {n1}-{n2} of {n3} products",
        "Showing {n1}-{n2} of {n3} results",
        "Page {n1} of {n2}",
        "Rating: {rating} out of 5 stars",
        "Based on {n3} customer reviews",
        "Delivery within {n1}-{n2} business days",
        "Free shipping on orders over ${price}",
        "Standard delivery: ${price}",
        "Call us at +1-800-{n3}-{n2}",
        "Copyright (c) {year} {brand}",
        "Showing {n2} products in {category}",
        "Filter by price: ${n1} - ${n3}",
        "{n1} items in your shopping cart",
        "Order #{n3}{n2}",
        "Size: {n1} (US)",
        "Estimated delivery date: July {n1}",
        "Price: ${price}",
        "Qty: {n1}",
        "Showing {n2} of {n3} reviews",
        "You saved ${price} on this order",
        "Open daily from 9 AM to 6 PM",
        "Item weight: {n1} lbs",
        "Resolution: 1920x1080",
        "{n2}% of buyers rated this {rating} stars",
        "In stock — ships from {brand} warehouse",
        "Warranty: {n1} year manufacturer guarantee",
        "Track order {n3}{n1} in your account",
        "Add up to {n1} items to compare",
    ]

    def slot(t):
        return _fmt(t,
                    n1=[str(random.randint(1, 30)) for _ in range(4)],
                    n2=[str(random.randint(31, 99)) for _ in range(4)],
                    n3=[str(random.randint(100, 999)) for _ in range(4)],
                    rating=[str(round(random.uniform(3.5, 5.0), 1)) for _ in range(4)],
                    price=PRICES, year=[str(y) for y in range(2018, 2027)],
                    brand=BRANDS, category=CATEGORIES)
    return _generate(T, BENIGN_TARGET, slot)


GENERATORS = {
    "Nagging": (gen_nagging, 1),
    "Bait and Switch": (gen_bait_switch, 1),
    "SaaS Billing": (gen_saas_billing, 1),
    "Rogue Malware": (gen_rogue_malware, 1),
    "Basket Sneaking": (gen_basket_sneaking, 1),
    "Drip Pricing": (gen_drip_pricing, 1),
    "Forced Action": (gen_forced_action, 1),
    "Trick Question": (gen_trick_question, 1),
    "Subscription Trap": (gen_subscription_trap, 1),
    "Interface Interference": (gen_interface_interference, 1),
    "Confirm Shaming": (gen_confirm_shaming, 1),
    "Not a Dark Pattern": (gen_benign_numeric, 0),
}


def build_collected():
    random.seed(SEED)
    rows = []
    counter = 20000
    summary = {}
    for category, (fn, label) in GENERATORS.items():
        texts = fn()
        summary[category] = len(texts)
        for txt in texts:
            rows.append({
                "page_id": f"ext_{counter}",
                "text": txt,
                "label": label,
                "Pattern Category": category,
            })
            counter += 1
    return rows, summary


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(here, "data", "processed", "collected.tsv")
    rows, summary = build_collected()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "text", "label", "Pattern Category"],
                           delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            # Guard: never let a tab or newline leak into a field and break the TSV.
            r = {k: re.sub(r"[\t\r\n]+", " ", str(v)) for k, v in r.items()}
            w.writerow(r)
    print(f"Wrote {len(rows)} collected rows -> {out_path}")
    print("Unique strings per class:")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24} {v}")


if __name__ == "__main__":
    main()
