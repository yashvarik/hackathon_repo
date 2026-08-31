
import os
import re
import json
import base64

from typing import TypedDict, Optional, Literal, Any, List

import fitz  # PyMuPDF

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import START, END, StateGraph


# ============================================================
# ENV
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env")


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="qwen/qwen3.8-27b",
    temperature=0,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)


# ============================================================
# TAVILY
# ============================================================

tavily_tool = TavilySearchResults(
    max_results=5
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):

    file_path: str

    images: Optional[List[str]]

    raw_output: Optional[str]

    parsed_output: Optional[dict]

    # FastAPI se ye value aayegi
    user_choice: Optional[
        Literal["jobs", "portfolio"]
    ]

    job_results: Optional[dict]

    portfolio_output: Optional[dict]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert resume parser.

You will receive images of a resume.

Analyze ALL provided resume pages carefully.

Extract only information that is actually present in the resume.

Return ONLY valid JSON.

Use exactly this structure:

{
    "name": null,
    "email": null,
    "phone": null,
    "location": null,
    "summary": null,
    "skills": [],
    "education": [],
    "experience": [],
    "projects": [],
    "certifications": [],
    "linkedin": null,
    "github": null
}

Rules:

1. Do not invent information.
2. If information is missing, return null.
3. Skills should be an array of strings.
4. Education should be an array of objects when possible.
5. Experience should be an array of objects when possible.
6. Projects should be an array of objects when possible.
7. Certifications should be an array.
8. Return ONLY JSON.
9. Do not use markdown.
10. Do not write explanations outside JSON.
"""


# ============================================================
# NODE 1
# PDF -> BASE64 IMAGES
# ============================================================

def chat(state: State):

    file_path = state["file_path"]

    if not os.path.exists(file_path):

        print("File does not exist:", file_path)

        return {
            "images": []
        }

    ext = os.path.splitext(file_path)[1].lower()

    if ext != ".pdf":

        print("Only PDF files are supported.")

        return {
            "images": []
        }

    doc = None

    try:

        doc = fitz.open(file_path)

        images = []

        for page in doc:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            img_bytes = pix.tobytes("png")

            img_base64 = base64.b64encode(
                img_bytes
            ).decode("utf-8")

            images.append(img_base64)

        # Maximum 3 pages
        images = images[:3]

        print(
            f"PDF loaded successfully. "
            f"Pages converted: {len(images)}"
        )

        return {
            "images": images
        }

    except Exception as e:

        print("PDF processing error:", e)

        return {
            "images": []
        }

    finally:

        if doc is not None:
            doc.close()


# ============================================================
# NODE 2
# IMAGES -> VISION LLM
# ============================================================

def img_raw_output(state: State):

    images = state.get("images") or []

    if not images:

        return {
            "raw_output": None
        }

    image_content = []

    # Text instruction
    image_content.append(
        {
            "type": "text",
            "text": """
Analyze all these resume pages and extract the candidate information.

Return ONLY valid JSON according to the required schema.
"""
        }
    )

    # Images
    for img_base64 in images:

        image_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64,"
                        + img_base64
                    )
                }
            }
        )

    system_message = SystemMessage(
        content=SYSTEM_PROMPT
    )

    human_message = HumanMessage(
        content=image_content
    )

    try:

        response = llm.invoke(
            [
                system_message,
                human_message
            ]
        )

        print("Vision model response received.")

        return {
            "raw_output": response.content
        }

    except Exception as e:

        print("Vision LLM error:", e)

        return {
            "raw_output": None
        }


# ============================================================
# HELPER
# CLEAN JSON
# ============================================================

def clean_json(text: str):

    if not text:
        return None

    text = text.strip()

    # Remove ```json
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove ```
    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


# ============================================================
# NODE 3
# RAW JSON -> PYTHON DICT
# ============================================================

def raw_parse(state: State):

    raw_output = state.get("raw_output")

    if not raw_output:

        return {
            "parsed_output": None
        }

    cleaned = clean_json(raw_output)

    try:

        parsed = json.loads(cleaned)

        print("Resume JSON parsed successfully.")

        return {
            "parsed_output": parsed
        }

    except json.JSONDecodeError as e:

        print("JSON parsing error:", e)

        print("Raw output:", raw_output)

        return {
            "parsed_output": None
        }


# ============================================================
# ROUTER
# ============================================================

def route_choice(state: State):

    choice = state.get("user_choice")

    print("User choice:", choice)

    if choice == "jobs":
        return "job_search"

    if choice == "portfolio":
        return "generate_portfolio"

    # Safety fallback
    print("Invalid choice. Defaulting to portfolio.")

    return "generate_portfolio"


# ============================================================
# NODE 4
# JOB SEARCH
# ============================================================

def job_search(state: State):

    parsed_resume = state.get(
        "parsed_output"
    )

    if not parsed_resume:

        return {
            "job_results": {
                "error": (
                    "Resume parsing failed. "
                    "Cannot search jobs."
                )
            }
        }

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    skills = parsed_resume.get(
        "skills"
    ) or []

    if not isinstance(skills, list):

        skills = []

    skills = [
        str(skill)
        for skill in skills
    ]

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    location = parsed_resume.get(
        "location"
    ) or ""

    # --------------------------------------------------------
    # Search query
    # --------------------------------------------------------

    search_query = (
        "entry level jobs for "
        + ", ".join(skills[:8])
        + " in "
        + str(location)
    )

    print("\nSearching jobs:")
    print(search_query)

    try:

        search_results = tavily_tool.invoke(
            {
                "query": search_query
            }
        )

    except Exception as e:

        print("Tavily error:", e)

        return {
            "job_results": {
                "error": "Job search failed."
            }
        }

    # --------------------------------------------------------
    # LLM prompt
    # --------------------------------------------------------

    system_prompt = """
You are an AI job recommendation assistant.

You will receive:

1. Candidate resume JSON
2. Job search results

Recommend the TOP 5 most relevant jobs.

Consider:

- Skills
- Experience
- Location
- Job title
- Required technologies
- Seniority

IMPORTANT:

Return ONLY valid JSON.

Use EXACTLY this structure:

{
    "jobs": [
        {
            "title": "",
            "company": "",
            "location": "",
            "experience_required": "",
            "reason": "",
            "match_score": 0
        }
    ]
}

Rules:

1. Maximum 5 jobs.
2. match_score must be between 0 and 100.
3. Do not invent companies.
4. Do not invent job listings.
5. Use only jobs supported by search results.
6. If experience requirement is higher than the candidate's experience, mention it.
7. Return JSON only.
"""

    human_prompt = f"""
RESUME:

{json.dumps(
    parsed_resume,
    indent=2
)}

JOB SEARCH RESULTS:

{json.dumps(
    search_results,
    indent=2,
    default=str
)}
"""

    try:

        response = llm.invoke(
            [
                SystemMessage(
                    content=system_prompt
                ),
                HumanMessage(
                    content=human_prompt
                )
            ]
        )

        cleaned = clean_json(
            response.content
        )

        jobs_json = json.loads(
            cleaned
        )

        print(
            "Job recommendations generated."
        )

        return {
            "job_results": jobs_json
        }

    except Exception as e:

        print(
            "Job recommendation error:",
            e
        )

        return {
            "job_results": {
                "error": (
                    "Failed to generate "
                    "job recommendations."
                ),
                "raw_output": (
                    response.content
                    if "response" in locals()
                    else None
                )
            }
        }


# ============================================================
# NODE 5
# PORTFOLIO
# ============================================================

def generate_portfolio(state: State):

    parsed_resume = state.get(
        "parsed_output"
    )

    if not parsed_resume:

        return {
            "portfolio_output": {
                "error": (
                    "Resume parsing failed."
                )
            }
        }

    system_prompt = """
You are an expert portfolio website content generator.

Given a resume JSON, generate professional portfolio content.

Return ONLY valid JSON.

Use EXACTLY this structure:

{
    "tagline": "",
    "about": "",
    "projects": [
        {
            "name": "",
            "description": "",
            "technologies": []
        }
    ]
}

Rules:

1. Do not invent information.
2. Improve the wording professionally.
3. Keep project descriptions concise.
4. Use technologies mentioned in the resume.
5. Return JSON only.
"""

    human_prompt = f"""
RESUME:

{json.dumps(
    parsed_resume,
    indent=2
)}
"""

    try:

        response = llm.invoke(
            [
                SystemMessage(
                    content=system_prompt
                ),
                HumanMessage(
                    content=human_prompt
                )
            ]
        )

        cleaned = clean_json(
            response.content
        )

        portfolio_json = json.loads(
            cleaned
        )

        print(
            "Portfolio generated successfully."
        )

        return {
            "portfolio_output":
                portfolio_json
        }

    except Exception as e:

        print(
            "Portfolio generation error:",
            e
        )

        return {
            "portfolio_output": {
                "error": (
                    "Failed to generate "
                    "portfolio."
                )
            }
        }


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(State)


graph.add_node(
    "chat",
    chat
)

graph.add_node(
    "img_raw",
    img_raw_output
)

graph.add_node(
    "raw_parse",
    raw_parse
)

graph.add_node(
    "job_search",
    job_search
)

graph.add_node(
    "generate_portfolio",
    generate_portfolio
)


# ============================================================
# EDGES
# ============================================================

graph.add_edge(
    START,
    "chat"
)

graph.add_edge(
    "chat",
    "img_raw"
)

graph.add_edge(
    "img_raw",
    "raw_parse"
)


# IMPORTANT:
# raw_parse ke baad directly user_choice ke basis
# par routing hogi.
graph.add_conditional_edges(
    "raw_parse",
    route_choice,
    {
        "job_search": "job_search",

        "generate_portfolio":
            "generate_portfolio"
    }
)


graph.add_edge(
    "job_search",
    END
)

graph.add_edge(
    "generate_portfolio",
    END
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = (
        r"C:\Users\yash\Downloads"
        r"\Yash_Varik_Resume.pdf"
    )

    # LOCAL TEST:
    # Yahan jobs ya portfolio choose kar sakta hai.
    choice = "jobs"

    result = app.invoke(
        {
            "file_path": pdf_path,

            "images": None,

            "raw_output": None,

            "parsed_output": None,

            "user_choice": choice,

            "job_results": None,

            "portfolio_output": None
        }
    )


    # ========================================================
    # RESUME JSON
    # ========================================================

    if result.get("parsed_output"):

        print("\n")
        print("=" * 60)

        print("RESUME JSON")

        print("=" * 60)

        print(
            json.dumps(
                result["parsed_output"],
                indent=4
            )
        )


    # ========================================================
    # JOBS
    # ========================================================

    if result.get("job_results"):

        print("\n")
        print("=" * 60)

        print("JOB RECOMMENDATIONS")

        print("=" * 60)

        print(
            json.dumps(
                result["job_results"],
                indent=4
            )
        )


    # ========================================================
    # PORTFOLIO
    # ========================================================

    if result.get("portfolio_output"):

        print("\n")
        print("=" * 60)

        print("PORTFOLIO")

        print("=" * 60)

        print(
            json.dumps(
                result["portfolio_output"],
                indent=4
            )
        )