# CREATECH Smart Match ML API

Separate FastAPI service for Smart Match semantic scoring.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8010
```

## Endpoint

```txt
POST /predict
```

Input:

```json
{
  "query": "I need a modern logo for a coffee shop",
  "candidates": [
    {
      "id": "1",
      "text": "Premium Logo Package Logo Design custom logo and brand kit"
    }
  ]
}
```

Output:

```json
{
  "scores": [
    {
      "id": "1",
      "score": 0.62
    }
  ]
}
```
