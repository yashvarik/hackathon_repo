import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from model import app

app_api = FastAPI()


TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@app_api.get("/")
def home():
    return {"message": "Resume Analyzer API is running"}


@app_api.post("/upload-resume")
async def upload_resume(
    resume: UploadFile = File(...),
    choice: str = Form(...)
):
    # -----------------------------
    # Validate choice
    # -----------------------------
    choice = choice.lower().strip()

    if choice not in ["jobs", "portfolio"]:
        raise HTTPException(
            status_code=400,
            detail="choice must be either 'jobs' or 'portfolio'"
        )

    # -----------------------------
    # Validate file
    # -----------------------------
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # -----------------------------
    # Save uploaded PDF
    # -----------------------------
    temp_path = os.path.join(
        TEMP_DIR,
        f"resume_{resume.filename}"
    )

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(resume.file, f)

        # -----------------------------
        # Run LangGraph
        # -----------------------------
        result = app.invoke({
            "file_path": temp_path,
            "user_choice": choice
        })

        # -----------------------------
        # Return only JSON-safe data
        # -----------------------------
        if choice == "jobs":

            return {
                "success": True,
                "choice": "jobs",
                "resume": result.get("parsed_output"),
                "jobs": result.get("job_results")
            }

        else:

            return {
                "success": True,
                "choice": "portfolio",
                "resume": result.get("parsed_output"),
                "portfolio": result.get("portfolio_output")
            }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # Delete temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)