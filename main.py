# app.py
import json
import time
from io import BytesIO
from typing import List, Dict, Any, Tuple

import streamlit as st
import requests
from pypdf import PdfReader

from google import genai
from google.genai.types import GenerateContentConfig
import streamlit as st
import hmac

def check_password() -> bool:
    """Returns True if the user is authenticated."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Already logged in
    if st.session_state.authenticated:
        return True

    # Read secrets
    try:
        correct_user = st.secrets["auth"]["username"]
        correct_pass = st.secrets["auth"]["password"]
    except Exception:
        st.error("Auth secrets not configured. Set [auth].username and [auth].password in secrets.")
        return False

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign in"):
        user_ok = hmac.compare_digest(username, correct_user)
        pass_ok = hmac.compare_digest(password, correct_pass)

        if user_ok and pass_ok:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect username or password")

    return False

# Gate the app
if not check_password():
    st.stop()
# -----------------------------
# Questions + batching
# -----------------------------
QUESTIONS = [
    "Is it easy to find and access the data (including the main bulletin, data, metadata, supporting information where available)?",
    "Is there a label showing the type of statistic and is the accredited official statistics badge shown where relevant?",
    "Are the statistics preannounced?",
    "Can you tell what date and time the statistics are released?",
    "Is it clear why these statistics are produced and what they should/shouldn’t be used for?",
    "Are the statistics described impartially?",
    "Are the main statistics summarised in clear and appropriate ways?",
    "Has the producer offered guidance on how to interpret or apply the data?",
    "Do the statistics include data or breakdowns on sensitive areas or topics such as on certain personal characteristics, contentious public issues, or topics such as suicide, abuse?",
    "Are multiple levels of granularity offered in the publication?",
    "Are links to fuller information about methods and quality provided?",
    "Is uncertainty presented in line with OSR guidance as either quantitative estimates or qualitative descriptions?",
    "Are statistical errors, such as bias, quantified and are measures of confidence produced?",
    "Are problems with comparing these statistics to others explained?",
    "Has the producer signposted other relevant statistical sources?",
    "Is there transparency about future changes to content?",
    "Has the producer made it easy to get in touch with the team?",
    "Has appropriate disclosure control methods been applied to the data before release?",
    "Needs work to incorporate other ways of thinking about accessibility (e.g. colour-blind friendly charts).",
    "Are the benefits and drawbacks of the data easy to understand?",
    "Is it clear what data sources have been used?",
    "Are existing data sources being used?",
    "Are connections to external datasets used to enhance the statistics?",
    "Is the impact of important limitations in the data sources explained?",
    "Are the analytical techniques and their rationale clearly described?",
    "Are the benefits and drawbacks of the statistics easy to understand?",
    "Is there transparency about problems affecting the latest statistics?",
    "Are the statistics consistent with relevant classifications and harmonised standards or is any legitimate departure from these standards clearly explained?",
    "Are there any planned changes to the methods and sources?",
    "Has the producer published a statement of compliance, within the statistical bulletin or centrally, outlining how the organisation and/or the statistics are produced in line with TQV?",
    "Has the producer published a release practice policy?",
    "Has the producer published a revisions policy?",
    "Has the producer explained and managed corrections well?",
    "Has a Pre-Release Access list been published?",
    "Has the producer published a data management policy?",
    "Is there clarity on how personal data is safeguarded?",
    "Has the producer published its quality management approach?",
    "Has the producer or producer team published an annual statistical work programme?",
    "Has the producer published a public involvement and engagement strategy?",
    "Is information on supplementary statistical services and related pricing policy available for users?",
    "Are users informed and supported of any data sharing and linkage opportunities for the underlying data?",
    "Do the press releases, social media posts, and other public statements from the producer appropriately and accurately interpret/represent the statistics?",
    "Is there any evidence that the producer is innovating or improving the statistics across TQV practices?",
    "Is there evidence that the producer sufficiently engages with a range of users and listens to their views?",
    "Is there any evidence to indicate the producer has collaborated with other producers or other stakeholders – across T or Q or V practices",
    "Is there any evidence to indicate that decisions about the statistics have been made impartially?",
    "Is there any mention of accuracy of the methods?",
]

TOOLS = [{"url_context": {}}]

# Fixed defaults (no UI controls)
MODEL_ID = "gemini-3-flash-preview"
BATCH_SIZE = 6
SLEEP_S = 0.0

# Practical limits to avoid massive prompts
PDF_MAX_PAGES = 60
PDF_MAX_CHARS_TOTAL = 180_000  # overall PDF text injected into prompt


def chunked(seq: List[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_client() -> genai.Client:
    return genai.Client(vertexai=True)


# -----------------------------
# PDF detection + extraction
# -----------------------------
def looks_like_pdf_url(url: str) -> bool:
    if url.lower().split("?")[0].endswith(".pdf"):
        return True
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        ctype = (r.headers.get("Content-Type") or "").lower()
        return "application/pdf" in ctype
    except Exception:
        return False


def extract_pdf_text_with_page_markers(url: str) -> str:
    """
    Downloads a PDF and returns text with page markers:
    [[PDF: <url> p.1]] ...text...
    """
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    reader = PdfReader(BytesIO(r.content))

    chunks: List[str] = []
    total_chars = 0

    num_pages = min(len(reader.pages), PDF_MAX_PAGES)
    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        marker = f"[[PDF: {url} p.{i+1}]]\n"
        piece = marker + text + "\n"

        # Cap total injected text to keep prompts manageable
        if total_chars + len(piece) > PDF_MAX_CHARS_TOTAL:
            remaining = max(PDF_MAX_CHARS_TOTAL - total_chars, 0)
            if remaining > 0:
                chunks.append(piece[:remaining])
            chunks.append(f"\n[[PDF: {url}]] (truncated due to size limits)\n")
            break

        chunks.append(piece)
        total_chars += len(piece)

    if not chunks:
        return f"[[PDF: {url}]] (No extractable text found)\n"

    return "\n\n".join(chunks)


def split_urls_into_html_and_pdf(urls: List[str]) -> Tuple[List[str], List[str]]:
    html_urls: List[str] = []
    pdf_urls: List[str] = []
    for u in urls:
        if looks_like_pdf_url(u):
            pdf_urls.append(u)
        else:
            html_urls.append(u)
    return html_urls, pdf_urls


# -----------------------------
# Batch LLM call (HTML via url_context + PDF via injected text)
# Now returns narrative answers + evidence (no Yes/No/Partially)
# -----------------------------
def answer_batch_with_sources(
    client: genai.Client,
    model_id: str,
    html_urls: List[str],
    pdf_text: str,
    questions: List[str],
    start_index: int,
    sleep_s: float = 0.0,
) -> Dict[str, Any]:
    numbered_questions = "\n".join([f"{start_index + i}. {q}" for i, q in enumerate(questions)])
    url_block = "\n".join(html_urls) if html_urls else "(none)"

    prompt = f"""
#CONTEXT#

You are a statistical analysis expert. You must answer each question using only:
- the url_context HTML sources (for the provided HTML URLs), and
- the supplied PDF text (authoritative; includes page markers).

No external knowledge, assumptions, or inference are allowed. If the sources do not state something, you must say so.

####

#OBJECTIVE#

For each question, produce:
1) a short, direct answer (1–4 sentences) that addresses the question using only the provided sources
2) evidence that is quoted or clearly referenced from the sources

If the information is not found, the answer must explicitly state it is not stated in the provided sources,
and the evidence must be exactly: "Not stated".

####

#STYLE#

British English. Precise, factual, evidence-driven. No speculation.

####

#RESPONSE#

Return ONLY valid JSON in the following schema (no markdown, no extra text):
{{
  "answers": [
    {{
      "q_number": <int>,
      "answer": <string>,
      "evidence": <string>
    }}
  ]
}}

Evidence rules:
- Quote short fragments or clearly reference where it appears.
- If evidence comes from a PDF, include the page marker like: [[PDF: … p.3]].
- If evidence comes from HTML sources, cite by describing the page/section clearly (e.g. "On the release page under 'Methodology'...").
- If not found: evidence MUST be exactly "Not stated".

####

HTML URLs to use via url_context:
{url_block}

PDF text:
{pdf_text}

Questions:
{numbered_questions}
""".strip()

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=GenerateContentConfig(tools=TOOLS),
    )

    # Join model text parts
    parts = []
    for p in response.candidates[0].content.parts:
        if getattr(p, "text", None):
            parts.append(p.text)
    raw = "".join(parts).strip()

    # Defensive JSON cleanup
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)

        # Attach question text + normalise fields
        for item in parsed.get("answers", []):
            idx = item["q_number"] - start_index
            if 0 <= idx < len(questions):
                item["question"] = questions[idx]

            if not item.get("answer"):
                item["answer"] = "Not stated in the provided sources."

            ev = item.get("evidence")
            if not ev:
                item["evidence"] = "Not stated"

        if sleep_s:
            time.sleep(sleep_s)

        return {"ok": True, "parsed": parsed, "raw": raw}

    except json.JSONDecodeError:
        return {"ok": False, "parsed": {"answers": []}, "raw": raw}


# -----------------------------
# Synthesis: overall feedback + improvements from the answered checklist
# (based ONLY on the answers/evidence already produced)
# -----------------------------
def synthesise_overall_report(
    client: genai.Client,
    model_id: str,
    answered_items: List[Dict[str, Any]],
    sleep_s: float = 0.0,
) -> Dict[str, Any]:
    # Keep payload compact (but still readable)
    compact = [
        {
            "q_number": x.get("q_number"),
            "question": x.get("question"),
            "answer": x.get("answer"),
            "evidence": x.get("evidence"),
        }
        for x in answered_items
    ]

    prompt = f"""
#CONTEXT#

You are reviewing a producer's publication against a TQV-style checklist.
You have a set of question-by-question answers and evidence. You MUST base your synthesis ONLY on that content.
Do not add new facts.

####

#OBJECTIVE#

Create an overall report in British English that includes:
- Overall summary (3–6 bullets)
- Strengths observed (bullets)
- Gaps / not stated areas (bullets)
- Prioritised improvements (High / Medium / Low) as bullets
- “Quick wins” (up to 6 bullets)
- Suggested additions to make the publication more auditable (e.g. clearer signposting, explicit links, release timings, methods/quality, uncertainty statements) BUT phrased generically unless explicitly supported

When you refer to an issue, you must ground it in the answered items (e.g. “Several items were not stated (Q3, Q4, Q11)…").
If most evidence is "Not stated", say so plainly.

####

#RESPONSE#

Return ONLY valid JSON in this schema:
{{
  "overall_report_markdown": <string>
}}

Input answered items (authoritative):
{json.dumps(compact, ensure_ascii=False, indent=2)}
""".strip()

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=GenerateContentConfig(),
    )

    parts = []
    for p in response.candidates[0].content.parts:
        if getattr(p, "text", None):
            parts.append(p.text)
    raw = "".join(parts).strip()

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        if not parsed.get("overall_report_markdown"):
            parsed["overall_report_markdown"] = "No overall report was generated."
        if sleep_s:
            time.sleep(sleep_s)
        return {"ok": True, "parsed": parsed, "raw": raw}
    except json.JSONDecodeError:
        return {"ok": False, "parsed": {"overall_report_markdown": "Failed to parse synthesis JSON output."}, "raw": raw}


def run_tqv_with_progress(urls: List[str], progress_cb) -> Tuple[str, Dict[str, Any]]:
    client = build_client()

    html_urls, pdf_urls = split_urls_into_html_and_pdf(urls)

    # Extract all PDFs up-front (with progress)
    pdf_text_blocks: List[str] = []
    if pdf_urls:
        for i, pdf_url in enumerate(pdf_urls, start=1):
            if progress_cb:
                progress_cb(0.02, f"Downloading & extracting PDF {i}/{len(pdf_urls)}...")
            try:
                pdf_text_blocks.append(extract_pdf_text_with_page_markers(pdf_url))
            except Exception as e:
                pdf_text_blocks.append(f"[[PDF: {pdf_url}]] (Failed to read PDF: {type(e).__name__})\n")

    pdf_text = "\n\n---\n\n".join(pdf_text_blocks) if pdf_text_blocks else "(no PDFs provided)"

    # Run question batches
    all_answers: List[Dict[str, Any]] = []
    q_num = 1
    batches = list(chunked(QUESTIONS, BATCH_SIZE))
    total_batches = len(batches)

    for b_idx, batch in enumerate(batches, start=1):
        if progress_cb:
            base = 0.05
            span = 0.80  # leave room for synthesis step
            pct = base + span * ((b_idx - 1) / max(total_batches, 1))
            progress_cb(pct, f"Answering questions batch {b_idx}/{total_batches}...")

        result = answer_batch_with_sources(
            client=client,
            model_id=MODEL_ID,
            html_urls=html_urls,
            pdf_text=pdf_text,
            questions=batch,
            start_index=q_num,
            sleep_s=SLEEP_S,
        )

        if result["ok"]:
            all_answers.extend(result["parsed"].get("answers", []))
        else:
            for i, q in enumerate(batch):
                all_answers.append(
                    {
                        "q_number": q_num + i,
                        "question": q,
                        "answer": "Not stated in the provided sources.",
                        "evidence": "Not stated",
                    }
                )

        q_num += len(batch)

    all_answers.sort(key=lambda x: x["q_number"])

    # Synthesis
    if progress_cb:
        progress_cb(0.90, "Generating overall report and prioritised improvements...")

    synth = synthesise_overall_report(
        client=client,
        model_id=MODEL_ID,
        answered_items=all_answers,
        sleep_s=SLEEP_S,
    )

    overall_md = synth["parsed"].get("overall_report_markdown", "No overall report was generated.")

    # Final Markdown report (questions + overall)
    lines = []
    lines.append("# TQV Checklist – Evidence-based Review\n")

    lines.append("## Question-by-question answers\n")
    for item in all_answers:
        lines.append(
            f"### Q{item['q_number']}. {item['question']}\n"
            f"**Answer:** {item.get('answer','')}\n\n"
            f"**Evidence:** {item.get('evidence','Not stated')}\n"
        )

    lines.append("\n---\n")
    lines.append("## Overall report\n")
    lines.append(overall_md.strip() + "\n")

    report_md = "\n".join(lines)

    payload = {
        "answers": all_answers,
        "overall_report_markdown": overall_md,
        "html_urls": html_urls,
        "pdf_urls": pdf_urls,
    }

    if progress_cb:
        progress_cb(1.0, "Done.")

    return report_md, payload


# -----------------------------
# Streamlit UI (outputs report only)
# -----------------------------
st.set_page_config(page_title="TQV Checklist", layout="wide")
st.title("TQV Checklist Automation")

urls_text = st.text_area(
    "Enter relevant URLs (one per line) — works with both web pages and PDF links",
    height=180,
    placeholder="https://...\nhttps://.../release.pdf",
)

run = st.button("Run")

if run:
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if not urls:
        st.error("Please enter at least one URL.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    def progress_cb(pct: float, msg: str):
        progress.progress(min(max(pct, 0.0), 1.0))
        status.info(msg)

    report_md, report_payload = run_tqv_with_progress(urls=urls, progress_cb=progress_cb)

    status.empty()
    progress.empty()

    st.subheader("Report")
    st.markdown(report_md)

    st.subheader("Download")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download JSON",
            data=json.dumps(report_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="tqv_report.json",
            mime="application/json",
        )
    with c2:
        st.download_button(
            "Download Markdown",
            data=report_md.encode("utf-8"),
            file_name="tqv_report.md",
            mime="text/markdown",
        )