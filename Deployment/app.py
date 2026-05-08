import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI  # Pakeista iš google.generativeai
from transformers import pipeline, AutoModelForSequenceClassification, RobertaTokenizer

# --- CONFIG ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # Pakeista aplinkos kintamojo pavadinimas
client = OpenAI(api_key=api_key) if api_key else None # Inicializuojamas OpenAI klientas

st.set_page_config(page_title="ReviewIntel", layout="wide", page_icon="📊")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0a0f !important; color: #f0f0f8 !important; font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] { background-color: #111118 !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }
[data-testid="stSidebar"] * { color: #f0f0f8 !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f0f0f8 !important; letter-spacing: -0.03em !important; }
.stButton > button { background: #6c63ff !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
.stButton > button:hover { background: #5a52dd !important; transform: translateY(-1px) !important; }
.stSelectbox > div > div { background: #1a1a26 !important; border: 1px solid rgba(255,255,255,0.12) !important; color: #f0f0f8 !important; border-radius: 8px !important; }
.stTextArea > div > div > textarea { background: #1a1a26 !important; border: 1px solid rgba(255,255,255,0.12) !important; color: #f0f0f8 !important; border-radius: 8px !important; }
.stMetric { background: #111118 !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 14px !important; padding: 1rem !important; }
[data-testid="stMetricLabel"] { color: #5a5a7a !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"] { color: #f0f0f8 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
.stProgress > div > div { background: #6c63ff !important; }
div[data-testid="stExpander"] { background: #111118 !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 14px !important; }
.stTabs [data-baseweb="tab-list"] { background: #1a1a26 !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #9090b0 !important; border-radius: 7px !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important; }
.stTabs [aria-selected="true"] { background: #6c63ff !important; color: white !important; }
[data-testid="stMarkdownContainer"] p { color: #9090b0 !important; }

.card { background: #111118; border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.25rem; margin-bottom: 1rem; }
.badge-pos { background: rgba(0,217,165,0.15); color: #00d9a5; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
.badge-neg { background: rgba(255,107,107,0.15); color: #ff6b6b; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
.badge-neu { background: rgba(255,209,102,0.15); color: #ffd166; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
.badge-cat { background: rgba(108,99,255,0.15); color: #6c63ff; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.hero-bar { background: linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(0,217,165,0.06) 100%); border: 1px solid rgba(108,99,255,0.2); border-radius: 14px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; }
.sentiment-bar-wrap { background: #1a1a26; border-radius: 20px; height: 10px; overflow: hidden; display: flex; margin: 6px 0; }
.product-rank { background: #1a1a26; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 1rem; margin: 0.5rem 0; display: flex; gap: 14px; align-items: flex-start; }
.review-card { background: #111118; border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; }
.summary-box { background: linear-gradient(135deg, #111118 0%, rgba(108,99,255,0.06) 100%); border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 1.5rem; margin-top: 1rem; }
.avoid-box { background: rgba(255,107,107,0.06); border: 1px solid rgba(255,107,107,0.2); border-radius: 14px; padding: 1.25rem; margin-top: 0.75rem; }
.action-box { background: rgba(0,217,165,0.06); border: 1px solid rgba(0,217,165,0.2); border-radius: 14px; padding: 1.25rem; margin-top: 0.75rem; }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: #00d9a5; display: inline-block; margin-right: 6px; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_sentiment_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_model")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.warning(f"Local model not found, using fallback: {e}")
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_pipe = load_sentiment_model()

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df_clusters = pd.read_csv("clustered_products.csv")
        df_reviews  = pd.read_csv("product_reviews.csv")
        df_clusters.columns = df_clusters.columns.str.strip()
        df_reviews.columns  = df_reviews.columns.str.strip()
        combined = pd.merge(df_reviews, df_clusters, on="name", how="inner")
        return combined
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None

df = load_data()

# --- HELPERS ---
def get_col(dataframe, options):
    for opt in options:
        if opt in dataframe.columns:
            return opt
    return None

def sentiment_bar_html(pos, neu, neg):
    return f"""
    <div class="sentiment-bar-wrap">
        <div style="width:{pos}%;background:#00d9a5;height:100%"></div>
        <div style="width:{neu}%;background:#ffd166;height:100%"></div>
        <div style="width:{neg}%;background:#ff6b6b;height:100%"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#5a5a7a;margin-top:2px">
        <span>✓ Pos {pos:.0f}%</span><span>~ Neu {neu:.0f}%</span><span>✗ Neg {neg:.0f}%</span>
    </div>"""

def generate_cluster_summary(cluster_name, reviews_text, top3_names, worst_name):
    """Call OpenAI and return structured marketing summary."""
    if not client:
        return None, "OpenAI API key missing. Add OPENAI_API_KEY to your .env file."

    prompt = f"""You are a senior marketing analyst. Analyze these customer reviews for the '{cluster_name}' product category.

Top 3 products by review volume: {', '.join(top3_names)}
Lowest rated product: {worst_name}

Sample customer reviews:
{reviews_text}

Provide a structured analysis with these EXACT section headers (use ## before each):

## TOP 3 PRODUCTS
For each of the 3 products listed above, write 2 sentences: one on key strengths customers mention, one on the main complaint.

## PRODUCT TO AVOID
Write 2-3 sentences explaining why '{worst_name}' underperforms based on the review patterns. Be specific about what customers dislike.

## MARKETING ACTION PLAN
Write exactly 4 bullet points starting with "-". Each bullet is one specific action the marketing team should take immediately. Cover: what to promote, what to fix in messaging, what negative theme to address, and what to stop doing.

Be direct, specific and professional. No generic advice."""

    try:
        # Pakeista į OpenAI API kvietimą su gpt-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional marketing analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
background:linear-gradient(135deg,#6c63ff,#00d9a5);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
margin-bottom:1.5rem;letter-spacing:-0.03em">ReviewIntel</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🔍 Classify a Review")
user_input = st.sidebar.text_area("Paste any review text:", height=120)

if st.sidebar.button("Classify Sentiment"):
    if user_input.strip():
        with st.sidebar:
            with st.spinner("Classifying..."):
                prediction = sentiment_pipe(user_input[:512])[0]
                label = prediction['label']
                score = prediction['score']
                color = "#00d9a5" if "pos" in label.lower() else "#ff6b6b" if "neg" in label.lower() else "#ffd166"
                st.markdown(f"""
                <div style="background:#1a1a26;border:1px solid rgba(255,255,255,0.12);
                border-radius:10px;padding:1rem;margin-top:0.5rem">
                    <div style="font-size:0.75rem;color:#5a5a7a;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:0.4rem">Result</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
                    color:{color}">{label.capitalize()}</div>
                    <div style="font-size:0.78rem;color:#9090b0;margin-top:4px">
                    Confidence: {score:.1%}</div>
                </div>""", unsafe_allow_html=True)
                st.progress(score)
    else:
        st.sidebar.warning("Please enter text first.")

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.75rem;color:#5a5a7a'>Amazon Electronics · NLP Pipeline</div>", unsafe_allow_html=True)

# --- STOP IF NO DATA ---
if df is None:
    st.error("Could not load CSV files. Make sure clustered_products.csv and product_reviews.csv are in the same folder.")
    st.stop()

# --- COLUMN DETECTION ---
rating_col   = get_col(df, ['reviews.rating_x', 'reviews.rating', 'rating'])
text_col     = get_col(df, ['reviews.text', 'text', 'full_review'])
user_col     = get_col(df, ['reviews.username', 'username', 'user'])
sent_col     = get_col(df, ['pred_label_name', 'vader_label_name', 'sentiment'])
cluster_list = sorted(df['cluster_name'].unique()) if 'cluster_name' in df.columns else []

total_reviews = len(df)
avg_rating    = df[rating_col].mean() if rating_col else 0.0

if sent_col:
    sc      = df[sent_col].str.lower().value_counts(normalize=True) * 100
    pos_pct = float(sc.get('positive', sc.get('pos', 0)))
    neg_pct = float(sc.get('negative', sc.get('neg', 0)))
    neu_pct = float(sc.get('neutral',  sc.get('neu', 0)))
else:
    pos_pct = neg_pct = neu_pct = 0.0

# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔎 Cluster Insights", "📝 Raw Data"])

# ══════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div class="hero-bar">
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
            letter-spacing:-0.03em;color:#f0f0f8">Marketing Intelligence Dashboard</div>
            <div style="font-size:0.85rem;color:#9090b0;margin-top:4px">
                Real-time product sentiment across all categories · <span class="live-dot"></span>Live
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{total_reviews:,}")
    col2.metric("Avg Rating",    f"{avg_rating:.1f} ★")
    col3.metric("Positive",      f"{pos_pct:.1f}%")
    col4.metric("Negative",      f"{neg_pct:.1f}%")

    st.markdown("<br><div style='font-family:Syne;font-weight:600;color:#f0f0f8'>Overall Sentiment Distribution</div>", unsafe_allow_html=True)
    st.markdown(sentiment_bar_html(pos_pct, neu_pct, neg_pct), unsafe_allow_html=True)

    st.markdown("<br><div style='font-family:Syne;font-weight:600;color:#f0f0f8'>Category Performance</div>", unsafe_allow_html=True)

    for cluster in cluster_list:
        cdata   = df[df['cluster_name'] == cluster]
        c_avg   = cdata[rating_col].mean() if rating_col else 0
        c_total = len(cdata)
        if sent_col:
            csc = cdata[sent_col].str.lower().value_counts(normalize=True) * 100
            cp  = float(csc.get('positive', 0))
            cn  = float(csc.get('neutral',  0))
            cne = float(csc.get('negative', 0))
        else:
            cp = cn = cne = 0.0
        overall  = "Positive" if cp > 60 else "Mixed" if cp > 40 else "Negative"
        badge_cl = "badge-pos" if cp > 60 else "badge-neu" if cp > 40 else "badge-neg"

        st.markdown(f"""
        <div class="review-card">
            <div style="display:flex;justify-content:space-between;align-items:center;
            margin-bottom:8px;flex-wrap:wrap;gap:6px">
                <span style="color:#f0f0f8;font-weight:600">{cluster}</span>
                <div style="display:flex;gap:8px;align-items:center">
                    <span style="font-size:0.75rem;color:#5a5a7a">{c_total:,} reviews</span>
                    <span style="color:#ffd166;font-weight:600">{c_avg:.1f} ★</span>
                    <span class="{badge_cl}">{overall}</span>
                </div>
            </div>
            {sentiment_bar_html(cp, cn, cne)}
        </div>""", unsafe_allow_html=True)

    # Top products table
    st.markdown("<br><div style='font-family:Syne;font-weight:600;color:#f0f0f8'>Top Rated Products</div>", unsafe_allow_html=True)
    if rating_col:
        top_prods = (df.groupby('name')
            .agg(avg_rating=(rating_col, 'mean'), review_count=(rating_col, 'count'),
                 cluster=('cluster_name', 'first'))
            .reset_index()
            .sort_values('avg_rating', ascending=False)
            .head(10))

        table_html = """<div class="card"><table style="width:100%;border-collapse:collapse;font-size:0.85rem">
        <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.07)">
            <th style="text-align:left;padding:10px 12px;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#5a5a7a">Product</th>
            <th style="text-align:left;padding:10px 12px;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#5a5a7a">Cluster</th>
            <th style="text-align:left;padding:10px 12px;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#5a5a7a">Rating</th>
            <th style="text-align:left;padding:10px 12px;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#5a5a7a">Reviews</th>
        </tr></thead><tbody>"""

        for _, row in top_prods.iterrows():
            short = str(row['name'])[:55] + '...' if len(str(row['name'])) > 55 else str(row['name'])
            table_html += f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
                <td style="padding:10px 12px;color:#f0f0f8;font-weight:500">{short}</td>
                <td style="padding:10px 12px"><span class="badge-cat">{row['cluster']}</span></td>
                <td style="padding:10px 12px;color:#ffd166;font-weight:600">{row['avg_rating']:.1f}★</td>
                <td style="padding:10px 12px;color:#9090b0">{int(row['review_count']):,}</td>
            </tr>"""
        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — CLUSTER INSIGHTS
# ══════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;
    letter-spacing:-0.04em;color:#f0f0f8;margin-bottom:0.4rem">
        Cluster <span style="color:#6c63ff">Insights</span>
    </div>
    <div style="font-size:0.9rem;color:#9090b0;margin-bottom:1.5rem">
        Select a cluster to explore products, reviews and AI summary
    </div>""", unsafe_allow_html=True)

    if "current_cluster" not in st.session_state:
        st.session_state.current_cluster = cluster_list[0] if cluster_list else ""

    btn_cols = st.columns(len(cluster_list))
    for i, c_name in enumerate(cluster_list):
        if btn_cols[i].button(c_name, key=f"btn_{c_name}"):
            st.session_state.current_cluster = c_name

    sel_cluster = st.session_state.current_cluster
    c_df        = df[df['cluster_name'] == sel_cluster]

    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:600;
    color:#f0f0f8;margin:1.5rem 0 1rem">
        Insights for: <span style="color:#6c63ff">{sel_cluster}</span>
    </div>""", unsafe_allow_html=True)

    # compute top 3 and worst before columns
    top3_names = c_df['name'].value_counts().head(3).index.tolist()

    if rating_col:
        prod_stats = (c_df.groupby('name')[rating_col]
                      .agg(['mean', 'count'])
                      .reset_index())
        prod_stats = prod_stats[prod_stats['count'] >= 3]
        worst_name = prod_stats.loc[prod_stats['mean'].idxmin(), 'name'] if not prod_stats.empty else "N/A"
    else:
        worst_name = "N/A"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""<div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;
        color:#5a5a7a;margin-bottom:0.75rem">🏆 Top 3 Products</div>""", unsafe_allow_html=True)

        for i, pname in enumerate(top3_names):
            short = str(pname)[:45] + '...' if len(str(pname)) > 45 else str(pname)
            st.markdown(f"""
            <div class="product-rank">
                <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                color:#6c63ff;min-width:28px;line-height:1">#{i+1}</div>
                <div style="font-size:0.85rem;color:#f0f0f8;font-weight:500">{short}</div>
            </div>""", unsafe_allow_html=True)

        if sent_col:
            csc = c_df[sent_col].str.lower().value_counts(normalize=True) * 100
            cp  = float(csc.get('positive', 0))
            cn  = float(csc.get('neutral',  0))
            cne = float(csc.get('negative', 0))
            st.markdown("<div style='margin-top:1rem;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;color:#5a5a7a;margin-bottom:0.5rem'>Sentiment</div>", unsafe_allow_html=True)
            st.markdown(sentiment_bar_html(cp, cn, cne), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="avoid-box" style="margin-top:1rem">
            <div style="font-size:0.72rem;font-weight:600;color:#ff6b6b;text-transform:uppercase;
            letter-spacing:0.06em;margin-bottom:0.4rem">⚠ Lowest Rated</div>
            <div style="font-size:0.82rem;color:#f0f0f8">{str(worst_name)[:50]}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;
        color:#5a5a7a;margin-bottom:0.75rem">📝 AI Summary + Marketing Actions (OpenAI)</div>""", unsafe_allow_html=True)

        if st.button("✨ Generate AI Summary", key="gen_summary"):
            if not client:
                st.error("API key missing. Add OPENAI_API_KEY to your .env file.")
            elif not text_col:
                st.error("Review text column not found in data.")
            else:
                with st.spinner("OpenAI is analyzing reviews..."):
                    txt_sample = "\n".join(
                        c_df[text_col].dropna().astype(str).head(15).tolist()
                    )
                    summary_text, error = generate_cluster_summary(
                        sel_cluster, txt_sample, top3_names, worst_name
                    )

                if error:
                    st.error(f"OpenAI error: {error}")
                else:
                    # split by ## and render each section
                    sections = summary_text.split("##")
                    for section in sections:
                        section = section.strip()
                        if not section:
                            continue

                        if section.upper().startswith("TOP 3"):
                            content = "\n".join(section.split("\n")[1:]).strip()
                            st.markdown(f"""
                            <div class="summary-box">
                                <div style="font-size:0.75rem;text-transform:uppercase;
                                letter-spacing:0.08em;color:#6c63ff;margin-bottom:0.75rem;font-weight:600">
                                🏆 Top 3 Products Analysis</div>
                                <div style="font-size:0.88rem;color:#9090b0;line-height:1.75">
                                {content.replace(chr(10), '<br>')}</div>
                            </div>""", unsafe_allow_html=True)

                        elif section.upper().startswith("PRODUCT TO AVOID"):
                            content = "\n".join(section.split("\n")[1:]).strip()
                            st.markdown(f"""
                            <div class="avoid-box">
                                <div style="font-size:0.75rem;text-transform:uppercase;
                                letter-spacing:0.08em;color:#ff6b6b;margin-bottom:0.75rem;font-weight:600">
                                ⚠ Product to Avoid</div>
                                <div style="font-size:0.88rem;color:#9090b0;line-height:1.75">
                                {content.replace(chr(10), '<br>')}</div>
                            </div>""", unsafe_allow_html=True)

                        elif section.upper().startswith("MARKETING ACTION PLAN"):
                            content = "\n".join(section.split("\n")[1:]).strip()
                            st.markdown(f"""
                            <div class="action-box">
                                <div style="font-size:0.75rem;text-transform:uppercase;
                                letter-spacing:0.08em;color:#00d9a5;margin-bottom:0.75rem;font-weight:600">
                                🎯 Marketing Action Plan</div>
                                <div style="font-size:0.88rem;color:#9090b0;line-height:1.75">
                                {content.replace(chr(10), '<br>')}</div>
                            </div>""", unsafe_allow_html=True)

    # recent reviews feed
    st.markdown("<div style='margin-top:2rem;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;color:#5a5a7a;margin-bottom:1rem'>Recent Reviews</div>", unsafe_allow_html=True)

    if text_col:
        for _, row in c_df.dropna(subset=[text_col]).head(8).iterrows():
            u      = str(row[user_col])[:20] if user_col else "Anonymous"
            r      = int(row[rating_col]) if rating_col and not pd.isna(row[rating_col]) else 0
            t      = str(row[text_col])[:200] + '...' if len(str(row[text_col])) > 200 else str(row[text_col])
            label  = str(row[sent_col]).lower() if sent_col else 'neutral'
            color  = "#00d9a5" if "pos" in label else "#ff6b6b" if "neg" in label else "#ffd166"
            stars  = '★' * r + '☆' * (5 - r)
            initials = u[:2].upper()

            st.markdown(f"""
            <div class="review-card">
                <div style="display:flex;gap:12px;align-items:flex-start">
                    <div style="width:36px;height:36px;border-radius:50%;background:rgba(108,99,255,0.15);
                    color:#6c63ff;display:flex;align-items:center;justify-content:center;
                    font-size:0.75rem;font-weight:700;flex-shrink:0">{initials}</div>
                    <div style="flex:1">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap">
                            <span style="font-size:0.85rem;font-weight:500;color:#f0f0f8">{u}</span>
                            <span style="color:{color};font-size:0.72rem;font-weight:600;
                            text-transform:uppercase;background:rgba(0,0,0,0.3);
                            padding:2px 8px;border-radius:20px">{label.capitalize()}</span>
                            <span style="color:#ffd166;font-size:0.75rem">{stars}</span>
                        </div>
                        <div style="font-size:0.85rem;color:#9090b0;line-height:1.6">{t}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3 — RAW DATA
# ══════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;
    letter-spacing:-0.04em;color:#f0f0f8;margin-bottom:0.4rem">
        Raw <span style="color:#00d9a5">Data</span>
    </div>
    <div style="font-size:0.9rem;color:#9090b0;margin-bottom:1.5rem">
        Filter and explore the full dataset
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        filter_cluster = st.selectbox("Filter by cluster:", ["All"] + cluster_list)
    with col2:
        if sent_col:
            sent_options = ["All"] + sorted(df[sent_col].dropna().unique().tolist())
            filter_sent  = st.selectbox("Filter by sentiment:", sent_options)
        else:
            filter_sent = "All"

    filtered = df.copy()
    if filter_cluster != "All":
        filtered = filtered[filtered['cluster_name'] == filter_cluster]
    if filter_sent != "All" and sent_col:
        filtered = filtered[filtered[sent_col] == filter_sent]

    st.markdown(f"<div style='font-size:0.82rem;color:#5a5a7a;margin-bottom:0.75rem'>Showing {len(filtered):,} rows</div>", unsafe_allow_html=True)

    show_cols = [c for c in ['name', 'cluster_name', text_col, rating_col, sent_col,
                              'prob_Negative', 'prob_Neutral', 'prob_Positive']
                 if c and c in filtered.columns]

    st.dataframe(
        filtered[show_cols].rename(columns={
            sent_col: 'sentiment',
            text_col: 'review_text'
        }).head(500),
        use_container_width=True,
        hide_index=True
    )