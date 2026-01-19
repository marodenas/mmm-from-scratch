FROM python:3.11-slim

WORKDIR /app

# Option A (simple): install from requirements.txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Option B (recommended later): install your package via pyproject
# COPY pyproject.toml README.md ./
# COPY src ./src
# RUN pip install --no-cache-dir -e ".[api]"

COPY . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
# To run Streamlit instead:
# CMD ["streamlit", "run", "src/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
