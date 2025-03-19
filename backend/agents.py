from gpt4all import GPT4All  # Using GPT4All for local NLP processing
from database import get_db
from models import LeaveRequestDB, CertificateRequestDB
from sqlalchemy.orm import Session

class LeaveRequestAgent:
    def __init__(self, db: Session):
        self.db = db

    def process_leave_request(self, leave_data):
        """Process leave requests by validating and storing them."""
        leave_entry = LeaveRequestDB(**leave_data)
        self.db.add(leave_entry)
        self.db.commit()
        return {"status": "approved", "message": "Leave request approved"}

class CertificateAgent:
    def __init__(self, db: Session):
        self.db = db

    def generate_certificate(self, cert_data):
        """Generate a certificate and store the request in the database."""
        cert_entry = CertificateRequestDB(**cert_data)
        self.db.add(cert_entry)
        self.db.commit()
        return {"status": "generated", "message": "Certificate issued"}

class QueryAgent:
    def __init__(self, model_path="gpt4all-model.bin"):
        """Initialize the query agent with a local NLP model."""
        self.model = GPT4All(model_path)

    def answer_query(self, query):
        """Process user queries using the AI model."""
        response = self.model.generate(query)
        return response

# Integrate with FastAPI
def get_agents(db: Session):
    return {
        "leave_agent": LeaveRequestAgent(db),
        "cert_agent": CertificateAgent(db),
        "query_agent": QueryAgent()
    }
