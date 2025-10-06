import streamlit as st
import pandas as pd
import joblib
import altair as alt
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIG ----------------
MODEL_PATH = "model.pkl"
MAILBOXLAYER_API_KEY = "72a7c8af17e55fe664aa3e3edae6ddee"
RAPID_KEY = "441676dba8msh7ee67ba6009fc37p1b47bbjsn90286de8ccfb"

MAILBOXLAYER_URL = "https://apilayer.net/api/check"
CRUNCHBASE_HEADERS = {
    "X-RapidAPI-Key": RAPID_KEY,
    "X-RapidAPI-Host": "crunchbase-crunchbase-v1.p.rapidapi.com"
}
B2B_HEADERS = {
    "X-RapidAPI-Key": RAPID_KEY,
    "X-RapidAPI-Host": "b2b-company-data-enrichment1.p.rapidapi.com"
}

st.set_page_config(page_title="AI Lead Validation & Enrichment", page_icon="🤖", layout="wide")




# -------------- MODEL LOAD --------------
@st.cache_data(show_spinner=False)
def load_model(path):
    return joblib.load(path)
model = load_model(MODEL_PATH)


# -------------- API HELPERS --------------
@st.cache_data(ttl=21600)
def validate_email_api(email):
    params = {"access_key": MAILBOXLAYER_API_KEY, "email": email, "smtp": 1, "format": 1}
    try:
        r = requests.get(MAILBOXLAYER_URL, params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400)
def get_company_permalink(company):
    try:
        r = requests.get(
            "https://crunchbase-crunchbase-v1.p.rapidapi.com/autocompletes",
            headers=CRUNCHBASE_HEADERS, params={"query": company}, timeout=8
        )
        if r.status_code == 200 and r.json().get("entities"):
            path = r.json()["entities"][0].get("path")
            if path and path.startswith("organization/"):
                return path.split("/")[-1]
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400)
def get_company_details(permalink):
    try:
        r = requests.get(
            f"https://crunchbase-crunchbase-v1.p.rapidapi.com/organizations/{permalink}",
            headers=CRUNCHBASE_HEADERS, timeout=8)
        if r.status_code == 200:
            return r.json().get("data", {}).get("organization", {}).get("properties", {})
    except Exception:
        pass
    return {}


@st.cache_data(ttl=86400)
def get_b2b_enrichment(domain):
    try:
        r = requests.get(
            "https://b2b-company-data-enrichment1.p.rapidapi.com/companies/enrich",
            headers=B2B_HEADERS, params={"domain": domain}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def map_to_status(r):
    if not r:
        return "unknown"
    s = r.get("score") or 0
    if r.get("smtp_check") and r.get("format_valid"):
        return "valid"
    if r.get("disposable") or s < 0.5:
        return "risky"
    if r.get("smtp_check") is False or r.get("format_valid") is False:
        return "invalid"
    return "unknown"


# ---------- SAFE FEATURE ENGINEERING ----------
def _safe_series(df, col, fill_value=""):
    return df[col].fillna(fill_value) if col in df.columns else pd.Series([fill_value]*len(df), index=df.index)


def preprocess_for_model(df, model):
    df = df.copy()
    linkedin_col = _safe_series(df, "linkedin", "")
    domain_col = _safe_series(df, "domain", "")
    df["linkedin_present"] = linkedin_col.astype(str).apply(lambda x: 1 if x.strip() else 0)
    df["professional_email"] = domain_col.astype(str).apply(
        lambda x: 0 if x.lower().endswith(("gmail.com", "yahoo.com", "hotmail.com")) else 1)
    df_feat = df.drop(["company", "email", "linkedin", "email_status", "status_icon"], axis=1, errors="ignore")
    if hasattr(model, "feature_names_in_"):
        df_feat = df_feat.reindex(columns=model.feature_names_in_, fill_value=0)
    for c in df_feat.columns:
        if df_feat[c].dtype == "object":
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").fillna(0)
    return df_feat


# ---------- ENRICH ----------
def enrich_lead(company, domain):
    out = {"company": company, "domain": domain, "linkedin": None, "employee_count": None,
           "total_funding": None, "category": None, "location": None, "industry": None}
    if company:
        perm = get_company_permalink(company)
        if perm:
            props = get_company_details(perm)
            out["linkedin"] = props.get("linkedin_url")
            emp = props.get("num_employees_enum")
            out["employee_count"] = int(emp) if str(emp).isdigit() else None
            out["total_funding"] = props.get("total_funding_usd")
            out["category"] = props.get("categories")
            out["location"] = props.get("city_name")
    if domain:
        b2b = get_b2b_enrichment(domain)
        if b2b:
            out["industry"] = b2b.get("industry", out["category"])
            emp = b2b.get("employee_count")
            out["employee_count"] = int(emp) if str(emp).isdigit() else out["employee_count"]
            out["location"] = b2b.get("location", out["location"])
    return out


# ---------- UI ----------
def main():
    

    st.markdown(
        """
        <div style='text-align:center; padding:20px;'>
            <h1 class='accent'>🤖 AI Lead Validation & Enrichment Dashboard</h1>
            <p>Smartly validate emails, enrich company data, and score leads — all in one dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if not upload:
        st.info("Upload a CSV file to start analysis.")
        return

    leads = pd.read_csv(upload)
    leads = leads.loc[:, ~leads.columns.str.contains("^Unnamed")]

    tab1, tab2, tab3, tab4 = st.tabs(["📧 Validate", "🏢 Enrich", "🤖 Score", "📤 Export"])

    # ---------------- EMAIL VALIDATION ----------------
    with tab1:
        st.subheader("📧 Email Validation")
        emails = leads.get("email", pd.Series([""] * len(leads)))
        results = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(validate_email_api, e.strip()): i for i, e in enumerate(emails)}
            progress = st.progress(0)
            for j, f in enumerate(as_completed(futures)):
                try:
                    res = f.result()
                except Exception:
                    res = None
                results.append(res)
                progress.progress((j + 1) / len(emails))
        leads["email_status"] = [map_to_status(r) for r in results]
        st.dataframe(leads[["email", "email_status"]])

    # ---------------- ENRICHMENT ----------------
    with tab2:
        st.subheader("🏢 Company Enrichment")
        st.info("Combining Crunchbase + B2B data sources for rich company insights.")
        enriched_records = []
        progress = st.progress(0)

        for i, row in leads.iterrows():
            company = str(row.get("company", "") or row.get("company_name", ""))
            domain = str(row.get("domain", "")) if "domain" in leads.columns else ""
            enriched_records.append(enrich_lead(company, domain))
            progress.progress((i + 1) / len(leads))
            time.sleep(0.02)

        enriched_df = pd.DataFrame(enriched_records)

        overlap = [c for c in enriched_df.columns if c in leads.columns]
        if overlap:
            enriched_df = enriched_df.drop(columns=overlap, errors="ignore")

        leads = pd.concat([leads.reset_index(drop=True),
                           enriched_df.reset_index(drop=True)], axis=1)

        leads = leads.loc[:, ~leads.columns.duplicated()]
        leads.columns = [c.strip() for c in leads.columns]

        st.success("✅ Enrichment complete — duplicates removed.")
        st.dataframe(leads.head())

    # ---------------- SCORING ----------------
    with tab3:
        st.subheader("🤖 AI Lead Scoring")
        X = preprocess_for_model(leads, model)
        scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        leads["Lead Score (%)"] = (scores * 100).round(2)
        penalty = {"valid": 1, "risky": 0.8, "invalid": 0.3, "unknown": 0.6}
        leads["Adjusted Score (%)"] = leads.apply(
            lambda r: r["Lead Score (%)"] * penalty.get(r["email_status"], 1), axis=1).round(2)
        st.bar_chart(leads["Adjusted Score (%)"])
        st.dataframe(leads[["email", "Lead Score (%)", "Adjusted Score (%)"]])

    # ---------------- EXPORT ----------------
    with tab4:
        csv = leads.to_csv(index=False).encode()
        st.download_button("📥 Download Enriched & Scored Leads", csv, "scored_leads.csv", "text/csv")
        st.success("All results ready for export!")

if __name__ == "__main__":
    main()
