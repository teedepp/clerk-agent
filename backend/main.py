from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from fastapi.responses import StreamingResponse
import requests
import logging
import io
from fpdf import FPDF

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)
DATABASE_URL = os.getenv("DATABASE_URL")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Check your .env file.")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Logging
logging.basicConfig(level=logging.INFO)

# Define Database Models
class LeaveRequestDB(Base):
    __tablename__ = "leave_requests"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, index=True)
    leave_type = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    reason = Column(String)

class CertificateRequestDB(Base):
    __tablename__ = "certificates"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    certificate_type = Column(String)

Base.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class LeaveRequest(BaseModel):
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    reason: str

class CertificateRequest(BaseModel):
    student_id: str
    certificate_type: str

class PromptRequest(BaseModel):
    messages: list

@app.get("/")
def read_root():
    return {"message": "Multi-Agent AI Backend is running"}

# Leave Management Endpoints
@app.post("/request_leave/")
def request_leave(leave: LeaveRequest, db: Session = Depends(get_db)):
    leave_entry = LeaveRequestDB(**leave.dict())
    db.add(leave_entry)
    db.commit()
    db.refresh(leave_entry)
    return {"message": "Leave request submitted successfully", "data": leave_entry}

@app.get("/leave_requests/")
def get_leave_requests(db: Session = Depends(get_db)):
    leaves = db.query(LeaveRequestDB).all()
    return {"message": "Leave requests retrieved successfully", "data": leaves}

# Certificate Management Endpoints
@app.post("/generate_certificate/")
def generate_certificate(cert: CertificateRequest, db: Session = Depends(get_db)):
    cert_entry = CertificateRequestDB(**cert.dict())
    db.add(cert_entry)
    db.commit()
    db.refresh(cert_entry)
    return {"message": "Certificate request submitted successfully", "certificate": cert_entry}

@app.get("/certificates/")
def get_certificates(db: Session = Depends(get_db)):
    certs = db.query(CertificateRequestDB).all()
    return {"message": "Certificates retrieved successfully", "data": certs}

@app.get("/download_certificate/{certificate_id}")
def download_certificate(certificate_id: int, db: Session = Depends(get_db)):
    cert = db.query(CertificateRequestDB).filter(CertificateRequestDB.id == certificate_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Certificate of {cert.certificate_type}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Issued to: {cert.student_id}", ln=True, align='C')

    # Save the file locally
    file_path = f"certificates/certificate_{certificate_id}.pdf"
    os.makedirs("certificates", exist_ok=True)  # Ensure the directory exists
    pdf.output(file_path)

    # Stream the file as response
    return StreamingResponse(open(file_path, "rb"), media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename=certificate_{certificate_id}.pdf"})





# Mistral AI Chat Endpoint
@app.post("/generate-response/")
def generate_response(request: PromptRequest):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistral-tiny",
        "messages": request.messages,
        "max_tokens": 150
    }

    response = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers)

    if response.status_code != 200:
        logging.error(f"Error from Mistral API: {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.json())

    response_json = response.json()
    
    # Check if 'choices' exist and extract the response properly
    if "choices" in response_json and len(response_json["choices"]) > 0:
        ai_message = response_json["choices"][0]["message"]["content"]
    else:
        ai_message = "No response from Mistral."

    return {"response": ai_message}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
