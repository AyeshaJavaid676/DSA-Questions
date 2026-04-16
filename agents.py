"""
FinOracle AI — Multi-Agent Implementations
Three agents: Supervisor, Extraction, Auditor.
Each agent is a pure function: FinOracleState → FinOracleState (partial update).
"""

import json
import logging
from typing import Any
from agents.state import FinOracleState
from agents.llm_client import call_llm
from tools.tools import FAISSVectorStore, tavily_search, analyze_image_with_qwen, run_financial_calculation
from src.config import FAISS_DIR, TOP_K_RETRIEVAL, CITATION_TEMPLATE

logger = logging.getLogger(__name__)

# Shared vector store instance
_vector_store: FAISSVectorStore | None = None

def get_vector_store() -> FAISSVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore(str(FAISS_DIR))
    return _vector_store


# ══════════════════════════════════════════════════════════════
# AGENT 1: SUPERVISOR
# Orchestrates the flow; decides which tools are needed.
# ══════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """You are a Financial Analysis Supervisor AI.
Your job is to analyze a user's financial question and determine the EXACT tools needed.

You must respond with a JSON object only. No preamble, no explanation.

JSON schema:
{
  "needs_rag_search": true/false,      // Search the uploaded PDF/document
  "needs_web_search": true/false,      // Search the web for context/news
  "needs_vision_ocr": true/false,      // Analyze charts/images in the PDF
  "needs_calculation": true/false,     // Perform mathematical verification
  "query_type": "factual|analytical|comparative|definitional",
  "key_metrics": ["list", "of", "metrics", "to", "extract"],
  "reasoning": "brief explanation of routing decision"
}

Rules:
- Always set needs_rag_search=true if a document is available
- Set needs_web_search=true for questions about industry context or recent news
- Set needs_vision_ocr=true for questions about charts, graphs, or visual tables
- Set needs_calculation=true for growth rates, margins, ratios, or any math
"""

def supervisor_node(state: FinOracleState) -> dict:
    """
    Route the query to appropriate agents.
    Returns partial state update with routing flags.
    """
    logger.info(f"[Supervisor] Routing query: {state['user_query'][:80]}...")

    mock_routing = json.dumps({
        "needs_rag_search": True,
        "needs_web_search": "context" in state["user_query"].lower() or "industry" in state["user_query"].lower(),
        "needs_vision_ocr": any(k in state["user_query"].lower() for k in ["chart", "graph", "visual", "image"]),
        "needs_calculation": any(k in state["user_query"].lower() for k in ["growth", "margin", "ratio", "calculate", "rate", "percent"]),
        "query_type": "analytical",
        "key_metrics": ["revenue", "net_income", "margins"],
        "reasoning": "Mock routing — PDF available, analytical question detected",
    })

    raw = call_llm(
        system_prompt=SUPERVISOR_SYSTEM,
        user_message=f"Question: {state['user_query']}\nPDF available: {bool(state.get('uploaded_pdf_path'))}",
        mock_response=mock_routing,
        mock=state["mock_mode"],
        temperature=0.0,
    )

    try:
        # Strip markdown fences if present
        clean = raw.strip().strip("```json").strip("```").strip()
        routing = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Supervisor] JSON parse failed, using defaults")
        routing = {"needs_rag_search": True, "needs_web_search": False,
                   "needs_vision_ocr": False, "needs_calculation": True}

    logger.info(f"[Supervisor] Routing decision: {routing}")

    return {
        "needs_web_search": routing.get("needs_web_search", False),
        "needs_vision_ocr": routing.get("needs_vision_ocr", False),
        "needs_calculation": routing.get("needs_calculation", True),
    }


# ══════════════════════════════════════════════════════════════
# AGENT 2: EXTRACTION AGENT
# Retrieves data from PDF (RAG + Vision) and Web.
# ══════════════════════════════════════════════════════════════

EXTRACTION_SYSTEM = """You are a Financial Data Extraction Specialist.
You extract precise financial data from document chunks and web search results.

CRITICAL RULES:
1. Only state facts that are explicitly in the provided context
2. Every number MUST be accompanied by its source citation
3. Use citation format: [Doc: {doc_name}, Page {page}, {section}]
4. If data is not found, say "Data not found in available sources"
5. Preserve exact numbers — do NOT round or estimate
6. Extract ALL relevant metrics for the user's question

Format your response as JSON:
{
  "extracted_facts": [
    {"metric": "Revenue FY2024", "value": "$94.4B", "citation": "[Doc: ..., Page X, Table Y]"},
    ...
  ],
  "data_found": true/false,
  "missing_data": ["list of metrics not found"],
  "raw_context_used": "brief description of sources used"
}
"""

def extraction_node(state: FinOracleState) -> dict:
    """
    Pull relevant data from FAISS index and web search.
    Returns extracted facts with citations.
    """
    logger.info("[Extraction] Retrieving relevant chunks...")

    # ── Step 1: RAG retrieval ─────────────────────────────────
    chunks = []
    if not state["mock_mode"]:
        try:
            vs = get_vector_store()
            chunks = vs.search(state["user_query"], top_k=TOP_K_RETRIEVAL)
        except FileNotFoundError:
            logger.warning("[Extraction] No FAISS index — skipping RAG")
    else:
        from src.mock_data import get_mock_chunks
        chunks = get_mock_chunks(state["user_query"])

    # ── Step 2: Web search (if flagged) ──────────────────────
    web_results = []
    if state.get("needs_web_search"):
        web_results = tavily_search(
            f"{state['user_query']} financial annual report",
            mock=state["mock_mode"],
        )

    # ── Step 3: Vision OCR (if flagged) ──────────────────────
    vision_outputs = []
    if state.get("needs_vision_ocr") and state.get("uploaded_pdf_path"):
        try:
            import fitz
            doc = fitz.open(state["uploaded_pdf_path"])
            # Analyze first 5 pages with charts
            for i, page in enumerate(doc[:5]):
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                ocr_result = analyze_image_with_qwen(img_bytes, mock=state["mock_mode"])
                vision_outputs.append(f"[Page {i+1} Vision]: {ocr_result}")
        except Exception as e:
            logger.error(f"[Extraction] Vision OCR failed: {e}")

    # ── Step 4: Extract structured data via LLM ──────────────
    context_parts = []
    citations_found = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        cite = CITATION_TEMPLATE.format(
            doc_name=meta.get("doc_name", "Unknown"),
            page=meta.get("page", "?"),
            section=meta.get("section", "General"),
        )
        context_parts.append(f"SOURCE {cite}:\n{chunk['text']}")
        citations_found.append(cite)

    for i, wr in enumerate(web_results):
        context_parts.append(f"WEB SOURCE [{wr.get('title', 'Web')}]:\n{wr.get('content', '')}")

    for vo in vision_outputs:
        context_parts.append(vo)

    full_context = "\n\n".join(context_parts) if context_parts else "No context available."

    mock_extraction = json.dumps({
        "extracted_facts": [
            {"metric": "Revenue FY2024", "value": "$94.4 billion", "citation": "[Doc: AnnualReport_2024.pdf, Page 14, Table 2 — Income Statement]"},
            {"metric": "Revenue FY2023", "value": "$84.1 billion", "citation": "[Doc: AnnualReport_2024.pdf, Page 14, Table 2 — Income Statement]"},
            {"metric": "Net Income FY2024", "value": "$21.9 billion", "citation": "[Doc: AnnualReport_2024.pdf, Page 15, Table 3 — Earnings Summary]"},
            {"metric": "Operating Margin FY2024", "value": "29.8%", "citation": "[Doc: AnnualReport_2024.pdf, Page 14, Table 2 — Income Statement]"},
        ],
        "data_found": True,
        "missing_data": [],
        "raw_context_used": "Income Statement tables, pages 14-15",
    })

    raw = call_llm(
        system_prompt=EXTRACTION_SYSTEM,
        user_message=f"Question: {state['user_query']}\n\nContext:\n{full_context}",
        mock_response=mock_extraction,
        mock=state["mock_mode"],
        temperature=0.0,
    )

    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        extracted = json.loads(clean)
    except Exception:
        extracted = {"extracted_facts": [], "data_found": False, "missing_data": ["Parse error"]}

    all_citations = [f["citation"] for f in extracted.get("extracted_facts", [])]

    return {
        "retrieved_chunks": chunks,
        "web_search_results": web_results,
        "vision_ocr_results": vision_outputs,
        "extracted_data": extracted,
        "citations": all_citations,
    }


# ══════════════════════════════════════════════════════════════
# AGENT 3: AUDITOR AGENT
# Verifies numbers via Python REPL. Produces the final answer.
# ══════════════════════════════════════════════════════════════

AUDITOR_SYSTEM = """You are a Financial Auditor AI with strict verification standards.

Your job:
1. Review extracted financial data
2. Generate Python code to verify every calculation (growth rates, margins, ratios)
3. Cross-check extracted numbers against each other for consistency
4. Produce a final, verified, well-cited answer

VERIFICATION CODE REQUIREMENTS:
- Print each calculation step with labels
- Store final results in a variable called `result`
- Flag any inconsistency with "⚠️ DISCREPANCY:"

CITATION REQUIREMENTS:
- Every number in your final answer MUST have [Doc: Page X, Table Y] citation
- If a number cannot be cited, mark it as [Unverified]

OUTPUT FORMAT:
{
  "verification_code": "python code string",
  "final_answer": "markdown formatted answer with citations",
  "audit_status": "PASSED|FAILED|PARTIAL",
  "confidence": 0.0-1.0
}
"""

def auditor_node(state: FinOracleState) -> dict:
    """
    Verify extracted data mathematically and produce the final answer.
    """
    logger.info("[Auditor] Verifying extracted data...")

    extracted = state.get("extracted_data", {})
    facts = extracted.get("extracted_facts", [])
    facts_str = json.dumps(facts, indent=2)

    mock_audit = json.dumps({
        "verification_code": (
            "# Auditor Verification\n"
            "rev_2024 = 94.4\n"
            "rev_2023 = 84.1\n"
            "growth_rate = (rev_2024 - rev_2023) / rev_2023 * 100\n"
            "print(f'Revenue YoY Growth: {growth_rate:.2f}%')\n"
            "net_income = 21.9\n"
            "net_margin = net_income / rev_2024 * 100\n"
            "print(f'Net Profit Margin: {net_margin:.2f}%')\n"
            "result = {'growth': growth_rate, 'net_margin': net_margin}\n"
        ),
        "final_answer": (
            "## FinOracle AI — Verified Financial Analysis\n\n"
            "### Revenue Performance\n"
            "FY2024 revenue reached **$94.4 billion**, a **+12.3% YoY increase** "
            "from $84.1B in FY2023. "
            "[Doc: AnnualReport_2024.pdf, Page 14, Table 2 — Income Statement]\n\n"
            "### Profitability Metrics\n"
            "| Metric | FY2024 | FY2023 | Change |\n"
            "|--------|--------|--------|--------|\n"
            "| Net Income | $21.9B | $18.7B | +17.1% |\n"
            "| Net Margin | 23.2% | 22.2% | +1.0 ppt |\n"
            "| Op. Margin | 29.8% | 28.4% | +1.4 ppt |\n\n"
            "[Doc: AnnualReport_2024.pdf, Page 15, Table 3 — Earnings Summary]\n\n"
            "### ✅ Audit Verification\n"
            "- Revenue growth: (94.4 − 84.1) / 84.1 × 100 = **12.25%** ✓\n"
            "- Net margin: 21.9 / 94.4 × 100 = **23.2%** ✓\n"
            "- All figures independently verified via Python REPL"
        ),
        "audit_status": "PASSED",
        "confidence": 0.95,
    })

    raw = call_llm(
        system_prompt=AUDITOR_SYSTEM,
        user_message=(
            f"User Question: {state['user_query']}\n\n"
            f"Extracted Facts:\n{facts_str}\n\n"
            "Please verify all calculations and produce the final answer."
        ),
        mock_response=mock_audit,
        mock=state["mock_mode"],
        temperature=0.0,
        max_tokens=3000,
    )

    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        audit_result = json.loads(clean)
    except Exception:
        audit_result = {
            "verification_code": "# Parse error",
            "final_answer": raw,
            "audit_status": "PARTIAL",
            "confidence": 0.5,
        }

    # ── Run the verification code ─────────────────────────────
    calc_log = []
    if audit_result.get("verification_code"):
        calc_result = run_financial_calculation(audit_result["verification_code"])
        if calc_result["error"]:
            calc_log.append(f"⚠️ Calculation error: {calc_result['error']}")
        else:
            calc_log.extend(calc_result["output"])
        logger.info(f"[Auditor] Calculations: {calc_log}")

    audit_passed = audit_result.get("audit_status") in ("PASSED", "PARTIAL")

    return {
        "final_answer": audit_result.get("final_answer", "No answer generated"),
        "confidence_score": audit_result.get("confidence", 0.5),
        "audit_passed": audit_passed,
        "calculation_log": calc_log,
    }
