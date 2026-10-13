
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PATIENT EXPERIENCE ANALYTICS DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Patient Experience Analytics",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- COLOURS -------------------------------

BG = "#0B1220"
PANEL = "#131E30"
PANEL_LIGHT = "#19283D"
NAVY = "#14243A"
TEAL = "#20C7B7"
TEAL_DARK = "#087F8C"
WHITE = "#F3F7FC"
MUTED = "#9AABC0"
RED = "#FF6577"
GREEN = "#42D6A4"
GOLD = "#F5C76B"
BORDER = "#283950"

# -------------------- PAGE STYLING --------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {WHITE};
    }}

    [data-testid="stHeader"] {{
        background: {BG};
    }}

    [data-testid="stSidebar"] {{
        background: #101A2A;
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: {WHITE};
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }}

    h1, h2, h3 {{
        color: {WHITE} !important;
    }}

    p, label, .stMarkdown {{
        color: {WHITE};
    }}

    div[data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 125px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEAL} !important;
    }}

    div[data-testid="stMetricDelta"] {{
        color: {GREEN} !important;
    }}

    div[data-testid="stTextArea"] textarea {{
        background: {PANEL};
        color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}

    .hero {{
        background: linear-gradient(115deg, #14243A 0%, #123746 100%);
        border: 1px solid #23576A;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 24px;
    }}

    .hero-label {{
        color: {TEAL};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .hero-title {{
        color: {WHITE};
        font-size: 34px;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        color: #C0CFDF;
        font-size: 15px;
    }}

    .section-heading {{
        font-size: 22px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 4px;
        color: {WHITE};
    }}

    .section-subtitle {{
        color: {MUTED};
        font-size: 13px;
        margin-bottom: 16px;
    }}

    .panel {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 20px;
    }}

    .small-label {{
        color: {MUTED};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .cause {{
        background: {PANEL_LIGHT};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 9px;
    }}

    .footer {{
        border-top: 1px solid {BORDER};
        padding-top: 18px;
        margin-top: 35px;
        color: {MUTED};
        font-size: 12px;
    }}

    .stButton > button {{
        background: {TEAL_DARK};
        color: white;
        border: 1px solid {TEAL};
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 20px;
        min-height: 44px;
    }}

    .stButton > button:hover {{
        background: {TEAL};
        color: #071522;
        border: 1px solid {TEAL};
    }}

    div[data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    button[data-baseweb="tab"] {{
        color: {MUTED};
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {TEAL};
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- CHART THEME ---------------------------

def style_chart(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=WHITE, family="Arial"),
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=WHITE),
        ),
        title_font=dict(size=16, color=WHITE),
        xaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=MUTED),
        ),
        yaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=MUTED),
        ),
    )
    return fig


# -------------------- LOAD DATA -----------------------------

BASE_DIR = Path(__file__).resolve().parent

def load_dataset():
    candidates = [
        BASE_DIR / "cleaned_hospital.csv",
        BASE_DIR / "hospital.csv",
        BASE_DIR / "preprocessed_hospital.csv",
    ]

    for file_path in candidates:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df.columns = [
                    str(c).strip() for c in df.columns
                ]
                df = df.loc[
                    :, ~df.columns.str.contains("^Unnamed")
                ]
                return df, file_path.name
            except Exception as exc:
                st.warning(
                    f"Could not read {file_path.name}: {exc}"
                )

    return pd.DataFrame(), None


df, dataset_name = load_dataset()


def find_column(columns, candidates):
    lower_map = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for column in columns:
        name = str(column).lower()
        if any(candidate.lower() in name for candidate in candidates):
            return column

    return None


TEXT_CANDIDATES = [
    "feedback", "review", "reviews", "patient_feedback",
    "patient review", "text", "comment", "comments",
    "processed_feedback", "cleaned_feedback",
]

SENTIMENT_CANDIDATES = [
    "sentiment", "sentiment_label", "label", "class",
    "polarity", "target",
]

RATING_CANDIDATES = [
    "rating", "ratings", "stars", "score",
]

text_column = None
sentiment_column = None
rating_column = None

if not df.empty:
    text_column = find_column(df.columns, TEXT_CANDIDATES)
    sentiment_column = find_column(
        df.columns, SENTIMENT_CANDIDATES
    )
    rating_column = find_column(
        df.columns, RATING_CANDIDATES
    )

    # If the review column has a different name, choose
    # the text-like column with the longest average content.
    if text_column is None:
        possible_text_columns = [
            c for c in df.columns
            if df[c].dtype == "object"
        ]

        if possible_text_columns:
            text_column = max(
                possible_text_columns,
                key=lambda c: df[c].astype(str).str.len().mean(),
            )


def normalize_sentiment(value):
    if pd.isna(value):
        return "Unknown"

    if isinstance(value, (int, float)):
        if value in (0, 1):
            return "Negative" if value == 0 else "Positive"
        if 1 <= value <= 2:
            return "Negative"
        if 3 <= value <= 5:
            return "Positive"

    value = str(value).strip().lower()

    if value in {"positive", "pos", "good", "favourable", "favorable"}:
        return "Positive"

    if value in {"negative", "neg", "bad", "unfavourable", "unfavorable"}:
        return "Negative"

    if value in {"neutral", "mixed"}:
        return "Neutral"

    if value in {"0", "0.0"}:
        return "Negative"

    if value in {"1", "1.0"}:
        return "Positive"

    # Handle numeric ratings stored as strings.
    try:
        number = float(value)
        if number in (0, 1):
            return "Negative" if number == 0 else "Positive"
        if 1 <= number <= 2:
            return "Negative"
        if 3 <= number <= 5:
            return "Positive"
    except ValueError:
        pass

    return "Unknown"


reviews = pd.DataFrame()

if not df.empty and text_column is not None:
    reviews["Review"] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reviews = reviews[reviews["Review"] != ""].copy()

    if sentiment_column is not None:
        reviews["Sentiment"] = df.loc[
            reviews.index, sentiment_column
        ].apply(normalize_sentiment)

    elif rating_column is not None:
        reviews["Sentiment"] = df.loc[
            reviews.index, rating_column
        ].apply(normalize_sentiment)

    else:
        reviews["Sentiment"] = "Unknown"

    reviews = reviews.reset_index(drop=True)


# -------------------- ASPECT DETECTION ---------------------

ASPECT_KEYWORDS = {
    "Waiting Time": [
        "wait", "waiting", "waited", "delay", "delayed",
        "queue", "long wait", "hours", "appointment",
    ],
    "Emergency": [
        "emergency", "ambulance", "emergency room",
        "emergency department",
    ],
    "Communication": [
        "communication", "communicate", "explained",
        "explanation", "informed", "listen", "ignored",
    ],
    "Facilities": [
        "facility", "facilities", "room", "bed",
        "equipment", "building",
    ],
    "Staff": [
        "staff", "nurse", "nurses", "receptionist",
        "reception", "employee",
    ],
    "Doctor": [
        "doctor", "doctors", "physician", "dr.",
    ],
    "Cleanliness": [
        "clean", "dirty", "toilet", "bathroom", "hygiene",
        "filthy", "unclean", "disgusting", "smelly", "stench",
    ],
    "Treatment": [
        "treatment", "care", "medicine", "medication",
        "drug", "pharmacy", "prescription", "surgery",
    ],
}

CAUSE_RULES = {
    "Long waiting times": [
        "waited", "waiting", "long wait", "long queue",
        "wait for hours", "hours to see", "delayed",
        "delay", "waited for hours",
    ],
    "Medication shortages": [
        "no medication", "no medicine", "out of stock",
        "ran out of medication", "ran out of medicine",
        "medication unavailable", "medicine unavailable",
        "not enough medication", "no drugs",
    ],
    "Dirty facilities and poor hygiene": [
        "disgusting", "filthy", "dirty", "unclean",
        "poor hygiene", "dirty toilet", "dirty bathroom",
        "smelly", "stinking",
    ],
    "Rude or unprofessional staff": [
        "rude staff", "staff were rude", "staff was rude",
        "rude nurse", "nurse was rude", "unprofessional staff",
        "disrespectful staff",
    ],
    "Poor doctor behaviour": [
        "rude doctor", "doctor was rude", "doctor is rude",
        "doctor ignored", "doctor did not listen",
        "doctor was unprofessional",
    ],
    "Poor communication": [
        "poor communication", "no communication",
        "did not explain", "not informed",
        "never explained", "ignored me",
    ],
    "Poor facilities or equipment": [
        "broken equipment", "poor facilities", "broken bed",
        "poor condition", "broken machine",
    ],
    "Poor treatment or care": [
        "poor treatment", "bad treatment", "wrong treatment",
        "poor care", "neglected", "neglect",
    ],
    "Emergency service problems": [
        "emergency delay", "emergency department",
        "ambulance never came", "emergency service",
    ],
}

NEGATIVE_WORDS = [
    "bad", "terrible", "worst", "rude", "dirty",
    "slow", "delay", "delayed", "poor", "unhelpful",
    "unprofessional", "horrible", "awful", "filthy",
    "unclean", "disrespectful", "disgusting", "missing",
    "unavailable", "shortage", "empty", "refused",
    "ignored", "neglected", "broken", "smelly",
    "not clean", "no medication", "no medicine",
]

POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "friendly",
    "helpful", "kind", "professional", "clean", "quick",
    "fast", "happy", "wonderful", "pleasant", "caring",
    "respectful", "efficient",
]


def phrase_present(text, phrase):
    # Flexible whitespace and word boundaries reduce accidental
    # matches inside unrelated words.
    pattern = r"(?<!\w)" + re.escape(phrase).replace(
        r"\ ", r"\s+"
    ) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def detect_aspects(text):
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(phrase_present(text, word) for word in keywords):
            found.append(aspect)
    return found


def aspect_sentiment(text, aspect):
    text_lower = text.lower()
    keywords = ASPECT_KEYWORDS.get(aspect, [])

    positions = []
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text_lower):
            positions.append(match.start())

    if not positions:
        return "Unknown"

    positive = 0
    negative = 0

    for position in positions:
        nearby = text_lower[
            max(0, position - 100): position + 150
        ]

        positive += sum(
            1 for word in POSITIVE_WORDS
            if phrase_present(nearby, word)
        )
        negative += sum(
            1 for word in NEGATIVE_WORDS
            if phrase_present(nearby, word)
        )

    if negative > positive:
        return "Negative"
    if positive > negative:
        return "Positive"

    return "Unknown"


def detect_causes(text):
    text_lower = text.lower()
    found = []

    for cause, keywords in CAUSE_RULES.items():
        if any(
            phrase_present(text_lower, keyword)
            for keyword in keywords
        ):
            found.append(cause)

    return found


def analyse_review_aspects(text):
    aspects = detect_aspects(text)
    results = []

    for aspect in aspects:
        results.append({
            "Aspect": aspect,
            "Aspect sentiment": aspect_sentiment(text, aspect),
        })

    return results


# -------------------- MODEL LOADING ------------------------

@st.cache_resource(show_spinner="Loading the DistilBERT model...")
def load_distilbert():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = BASE_DIR / "distilbert_patient_model"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model folder not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )
    model.eval()

    return tokenizer, model, torch


def predict_review(text):
    tokenizer, model, torch = load_distilbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits, dim=1
        )[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())

    # Use the model's saved label mapping if available.
    id2label = getattr(model.config, "id2label", {}) or {}
    saved_label = str(
        id2label.get(predicted_id, "")
    ).strip().lower()

    if "neg" in saved_label:
        sentiment = "Negative"
    elif "pos" in saved_label:
        sentiment = "Positive"
    else:
        # Standard binary classification convention used
        # by this project's trained model: 0 = negative, 1 = positive.
        sentiment = "Positive" if predicted_id == 1 else "Negative"

    return sentiment, confidence


# -------------------- PREPARE ANALYTICS ---------------------

total_reviews = len(reviews)

positive_count = int(
    (reviews["Sentiment"] == "Positive").sum()
) if total_reviews else 0

negative_count = int(
    (reviews["Sentiment"] == "Negative").sum()
) if total_reviews else 0

neutral_count = int(
    (reviews["Sentiment"] == "Neutral").sum()
) if total_reviews else 0

unknown_count = int(
    (reviews["Sentiment"] == "Unknown").sum()
) if total_reviews else 0

known_count = positive_count + negative_count + neutral_count

positive_pct = (
    positive_count / known_count * 100
    if known_count else 0
)

negative_pct = (
    negative_count / known_count * 100
    if known_count else 0
)

aspect_rows = []
cause_counts = {}

for review in reviews["Review"].tolist() if total_reviews else []:
    for aspect_result in analyse_review_aspects(review):
        aspect_rows.append(aspect_result)

    for cause in detect_causes(review):
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

aspect_df = pd.DataFrame(
    aspect_rows,
    columns=["Aspect", "Aspect sentiment"],
)

if not aspect_df.empty:
    aspect_summary = (
        aspect_df.groupby("Aspect")
        .size()
        .reset_index(name="Mentions")
        .sort_values("Mentions", ascending=False)
    )

    negative_aspect_summary = (
        aspect_df[aspect_df["Aspect sentiment"] == "Negative"]
        .groupby("Aspect")
        .size()
        .reset_index(name="Negative mentions")
        .sort_values("Negative mentions", ascending=False)
    )
else:
    aspect_summary = pd.DataFrame(
        columns=["Aspect", "Mentions"]
    )
    negative_aspect_summary = pd.DataFrame(
        columns=["Aspect", "Negative mentions"]
    )

cause_df = pd.DataFrame(
    [
        {"Potential problem area": cause, "Mentions": count}
        for cause, count in cause_counts.items()
    ],
    columns=["Potential problem area", "Mentions"],
)

if not cause_df.empty:
    cause_df = cause_df.sort_values(
        "Mentions", ascending=False
    )


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Hospital intelligence · NLP analytics</div>
        <div class="hero-title">Patient Experience Analytics</div>
        <div class="hero-subtitle">
            Understand patient sentiment, explore recurring concerns,
            and identify potential areas for quality improvement.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- SIDEBAR -------------------------------

with st.sidebar:
    st.markdown("## ✚ Patient Analytics")
    st.caption("NLP-powered decision support")

    st.markdown("---")
    st.markdown("### Dashboard sections")
    st.markdown(
        """
        - Overview
        - Sentiment analysis
        - Model performance
        - Aspect analysis
        - Potential root causes
        - Individual review analysis
        """
    )

    st.markdown("---")
    st.markdown("### Dataset status")

    if dataset_name:
        st.success(f"Loaded: {dataset_name}")
        st.write(f"**Usable reviews:** {total_reviews}")
    else:
        st.error("No hospital CSV file found.")

    if not df.empty:
        st.write(f"**Detected review column:** {text_column}")
        st.write(f"**Detected label column:** {sentiment_column or rating_column or 'None'}")

    st.markdown("---")
    st.caption(
        "Potential root causes are detected using text patterns. "
        "They require human review and are not confirmed causes."
    )


# ============================================================
# 1. OVERVIEW METRICS
# ============================================================

st.markdown(
    '<div class="section-heading">Overview</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">A snapshot of the available patient feedback dataset.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total reviews", f"{total_reviews:,}")

with m2:
    st.metric(
        "Positive reviews",
        f"{positive_count:,}",
        f"{positive_pct:.1f}% of labelled reviews",
    )

with m3:
    st.metric(
        "Negative reviews",
        f"{negative_count:,}",
        f"{negative_pct:.1f}% of labelled reviews",
    )

with m4:
    st.metric(
        "Potential problem areas",
        f"{len(cause_df):,}",
        "Types detected in the dataset",
    )

if total_reviews == 0:
    st.warning(
        "The dashboard could not find usable reviews. "
        "Check that hospital.csv or cleaned_hospital.csv is in "
        "the same folder as final_dashboard.py and contains a "
        "patient-feedback text column."
    )


# ============================================================
# 2. SENTIMENT ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Sentiment analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Distribution of sentiment labels available in the dataset.</div>',
    unsafe_allow_html=True,
)

sentiment_col, sentiment_info_col = st.columns([1.15, 1])

with sentiment_col:
    if known_count > 0:
        sentiment_chart_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Reviews": [
                positive_count, negative_count, neutral_count
            ],
        })
        sentiment_chart_data = sentiment_chart_data[
            sentiment_chart_data["Reviews"] > 0
        ]

        fig_sentiment = px.pie(
            sentiment_chart_data,
            names="Sentiment",
            values="Reviews",
            hole=0.68,
            color="Sentiment",
            color_discrete_map={
                "Positive": GREEN,
                "Negative": RED,
                "Neutral": GOLD,
            },
        )
        fig_sentiment.update_traces(
            textposition="inside",
            textinfo="percent",
            marker=dict(line=dict(color=PANEL, width=3)),
        )
        fig_sentiment.update_layout(
            annotations=[
                dict(
                    text=f"{known_count:,}<br>labelled",
                    x=0.5,
                    y=0.5,
                    font=dict(size=19, color=WHITE),
                    showarrow=False,
                )
            ]
        )
        st.plotly_chart(
            style_chart(fig_sentiment, 340),
            use_container_width=True,
        )
    else:
        st.info(
            "No recognisable sentiment labels were found in the CSV. "
            "The review input below can still use DistilBERT."
        )

with sentiment_info_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Sentiment summary")

    st.markdown(
        f"""
        <p style="color:{MUTED};">Positive</p>
        <h2 style="color:{GREEN};">{positive_count:,}</h2>
        <p style="color:{MUTED};">Negative</p>
        <h2 style="color:{RED};">{negative_count:,}</h2>
        <p style="color:{MUTED};">Unlabelled or unrecognised</p>
        <h3 style="color:{GOLD};">{unknown_count:,}</h3>
        """,
        unsafe_allow_html=True,
    )

    if known_count:
        st.progress(min(max(positive_pct / 100, 0), 1))
        st.caption(
            f"{positive_pct:.1f}% positive among recognised labels"
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 3. MODEL PERFORMANCE COMPARISON
# ============================================================

st.markdown(
    '<div class="section-heading">Model performance</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Previously recorded evaluation results for the three project models.</div>',
    unsafe_allow_html=True,
)

model_metrics = pd.DataFrame([
    {
        "Model": "VADER",
        "Accuracy": 86,
        "Precision": 89,
        "Recall": 93,
        "F1-score": 91,
    },
    {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": 81,
        "Precision": 80,
        "Recall": 99,
        "F1-score": 88,
    },
    {
        "Model": "DistilBERT",
        "Accuracy": 91,
        "Precision": 95,
        "Recall": 93,
        "F1-score": 94,
    },
])

model_chart = model_metrics.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score (%)",
)

model_col, model_table_col = st.columns([1.3, 1])

with model_col:
    fig_models = px.bar(
        model_chart,
        x="Model",
        y="Score (%)",
        color="Metric",
        barmode="group",
        color_discrete_sequence=[
            TEAL, "#648DCE", GOLD, GREEN
        ],
        text="Score (%)",
    )
    fig_models.update_traces(textposition="outside")
    fig_models.update_yaxes(range=[0, 110])
    st.plotly_chart(
        style_chart(fig_models, 390),
        use_container_width=True,
    )

with model_table_col:
    st.markdown("#### Evaluation metrics")
    display_metrics = model_metrics.copy()
    for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_metrics[metric] = (
            display_metrics[metric].astype(str) + "%"
        )

    st.dataframe(
        display_metrics,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div class="panel">
            <div class="small-label">Best overall in the recorded evaluation</div>
            <h2 style="color:{TEAL};">DistilBERT</h2>
            <p style="color:{MUTED};">
                Accuracy: 91% &nbsp; · &nbsp; F1-score: 94%
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "These are the project's previously recorded evaluation metrics, "
    "not measurements recalculated from the currently loaded CSV."
)


# ============================================================
# 4. ASPECT ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Aspect analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Which aspects of the patient experience appear in the feedback?</div>',
    unsafe_allow_html=True,
)

aspect_col, negative_aspect_col = st.columns(2)

with aspect_col:
    st.markdown("#### Most-mentioned aspects")

    if not aspect_summary.empty:
        fig_aspects = px.bar(
            aspect_summary.sort_values("Mentions"),
            x="Mentions",
            y="Aspect",
            orientation="h",
            text="Mentions",
            color="Mentions",
            color_continuous_scale=[
                "#173447", TEAL
            ],
        )
        fig_aspects.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            style_chart(fig_aspects, 380),
            use_container_width=True,
        )
    else:
        st.info("No aspects were detected in the available reviews.")

with negative_aspect_col:
    st.markdown("#### Aspects with negative wording")

    if not negative_aspect_summary.empty:
        fig_negative_aspects = px.bar(
            negative_aspect_summary.sort_values(
                "Negative mentions"
            ),
            x="Negative mentions",
            y="Aspect",
            orientation="h",
            text="Negative mentions",
            color_discrete_sequence=[RED],
        )
        st.plotly_chart(
            style_chart(fig_negative_aspects, 380),
            use_container_width=True,
        )
    else:
        st.info(
            "No negative aspect wording was detected by the "
            "keyword-based aspect analysis."
        )

st.caption(
    "Aspect sentiment is estimated using nearby positive and negative "
    "keywords. It is a heuristic and may miss context, sarcasm, or negation."
)


# ============================================================
# 5. POTENTIAL ROOT-CAUSE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Potential root-cause analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Recurring complaint patterns that hospital management may wish to investigate.</div>',
    unsafe_allow_html=True,
)

cause_chart_col, cause_list_col = st.columns([1.15, 1])

with cause_chart_col:
    st.markdown("#### Potential problem areas across reviews")

    if not cause_df.empty:
        fig_causes = px.bar(
            cause_df.sort_values("Mentions"),
            x="Mentions",
            y="Potential problem area",
            orientation="h",
            text="Mentions",
            color_discrete_sequence=[TEAL],
        )
        st.plotly_chart(
            style_chart(fig_causes, 390),
            use_container_width=True,
        )
    else:
        st.info(
            "No matching root-cause keywords were found. "
            "This does not mean there are no problems."
        )

with cause_list_col:
    st.markdown("#### Detected complaint patterns")

    if not cause_df.empty:
        for _, row in cause_df.iterrows():
            st.markdown(
                f"""
                <div class="cause">
                    <div style="font-weight:650;color:{WHITE};">
                        {row['Potential problem area']}
                    </div>
                    <div style="color:{MUTED};font-size:13px;margin-top:4px;">
                        Detected in {int(row['Mentions'])} review(s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write(
            "Complaint patterns will appear here when matching "
            "phrases are found in the dataset."
        )

st.warning(
    "These are potential problem areas identified through keywords, "
    "not verified root causes. Counts can overlap because one review "
    "may mention several problems."
)


# ============================================================
# 6. INDIVIDUAL PATIENT REVIEW ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Analyse an individual patient review</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Enter feedback to predict sentiment and inspect aspects and possible complaint causes.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    review_text = st.text_area(
        "Patient feedback",
        placeholder=(
            "Example: I waited for three hours, the doctor was rude, "
            "and there was no medication available."
        ),
        height=140,
        key="individual_review",
    )

    analyse_col, clear_col, _ = st.columns([1, 1, 3])

    with analyse_col:
        analyse_clicked = st.button(
            "✦ Analyse review",
            use_container_width=True,
        )

    with clear_col:
        clear_clicked = st.button(
            "Clear input",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state["individual_review"] = ""
        st.rerun()

    if analyse_clicked:
        if not review_text.strip():
            st.warning("Please enter a patient review first.")
        else:
            try:
                with st.spinner("Analysing the review with DistilBERT..."):
                    predicted_sentiment, confidence = predict_review(
                        review_text.strip()
                    )

                st.markdown("---")
                st.markdown("### Analysis results")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    sentiment_colour = (
                        GREEN if predicted_sentiment == "Positive"
                        else RED
                    )
                    st.markdown(
                        f"""
                        <div class="panel">
                            <div class="small-label">Predicted sentiment</div>
                            <h1 style="color:{sentiment_colour};">
                                {predicted_sentiment}
                            </h1>
                            <p style="color:{MUTED};">
                                DistilBERT classification
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with result_col2:
                    st.markdown(
                        f"""
                        <div class="panel">
                            <div class="small-label">Model confidence</div>
                            <h1 style="color:{TEAL};">
                                {confidence * 100:.2f}%
                            </h1>
                            <p style="color:{MUTED};">
                                Confidence in the predicted class,
                                not overall model accuracy.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.progress(min(max(confidence, 0), 1))

                found_aspects = analyse_review_aspects(review_text)
                found_causes = detect_causes(review_text)

                aspect_result_col, cause_result_col = st.columns(2)

                with aspect_result_col:
                    st.markdown("#### Detected aspects")

                    if found_aspects:
                        result_aspect_df = pd.DataFrame(found_aspects)
                        st.dataframe(
                            result_aspect_df,
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            "No known aspect keywords were detected."
                        )

                with cause_result_col:
                    st.markdown("#### Potential problem areas")

                    if found_causes:
                        for cause in found_causes:
                            st.markdown(
                                f"""
                                <div class="cause">
                                    <span style="color:{RED};">●</span>
                                    <span style="color:{WHITE};">
                                        {cause}
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info(
                            "No listed problem pattern was detected."
                        )

                st.markdown("#### Review summary")

                st.write(review_text)

                if predicted_sentiment == "Negative":
                    st.error(
                        "This review was classified as negative. "
                        "The detected aspects and complaint patterns "
                        "may help guide further investigation."
                    )
                else:
                    st.success(
                        "This review was classified as positive. "
                        "Check the detected aspects for additional context."
                    )

                st.caption(
                    "Sentiment is predicted by DistilBERT. Aspect and "
                    "problem-area detection uses keyword rules and can "
                    "make mistakes. Results should be reviewed by a person."
                )

            except Exception as exc:
                st.error(
                    "The review could not be analysed. Check that the "
                    "DistilBERT model folder exists and that its required "
                    "packages are installed."
                )
                st.code(str(exc))


# ============================================================
# 7. EXPLORE DATASET
# ============================================================

st.markdown(
    '<div class="section-heading">Explore patient feedback</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Inspect the available records and filter by sentiment.</div>',
    unsafe_allow_html=True,
)

if total_reviews > 0:
    filter_options = [
        "All",
        "Positive",
        "Negative",
        "Neutral",
        "Unknown",
    ]

    selected_filter = st.selectbox(
        "Filter reviews",
        filter_options,
        index=0,
    )

    filtered_reviews = reviews.copy()

    if selected_filter != "All":
        filtered_reviews = filtered_reviews[
            filtered_reviews["Sentiment"] == selected_filter
        ]

    st.caption(
        f"Showing {len(filtered_reviews):,} of {total_reviews:,} reviews."
    )

    st.dataframe(
        filtered_reviews,
        hide_index=True,
        use_container_width=True,
        height=300,
    )

    csv_data = filtered_reviews.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download displayed reviews as CSV",
        data=csv_data,
        file_name="patient_feedback_analysis.csv",
        mime="text/csv",
    )


# -------------------- FOOTER -------------------------------

st.markdown(
    """
    <div class="footer">
        <b>Patient Experience Analytics</b><br>
        NLP-based decision-support prototype for patient feedback analysis.
        This system does not replace clinical judgement, formal investigations,
        or human review.
    </div>
    """,
    unsafe_allow_html=True,
)

