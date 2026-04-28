import uuid
import json
import os
import pickle

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "chicago_business_survival_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LICENSE_TYPES = [
    "Consumption on Premises - Incidental Activity",
    "Home Occupation",
    "Home Repair",
    "Limited Business License",
    "Manufacturing Establishments",
    "Package Goods",
    "Regulated Business License",
    "Retail Food Establishment",
    "Tavern",
    "Tobacco",
]

NEIGHBORHOODS = {
    "the loop": (41.8827, -87.6233),
    "loop": (41.8827, -87.6233),
    "downtown": (41.8827, -87.6233),
    "river north": (41.8916, -87.6328),
    "wicker park": (41.9084, -87.6790),
    "lincoln park": (41.9242, -87.6654),
    "logan square": (41.9217, -87.7048),
    "pilsen": (41.8540, -87.6601),
    "bridgeport": (41.8332, -87.6451),
    "hyde park": (41.7943, -87.5907),
    "uptown": (41.9653, -87.6579),
    "andersonville": (41.9810, -87.6678),
    "lakeview": (41.9434, -87.6494),
    "wrigleyville": (41.9484, -87.6553),
    "old town": (41.9103, -87.6368),
    "gold coast": (41.9038, -87.6268),
    "streeterville": (41.8930, -87.6193),
    "south loop": (41.8662, -87.6247),
    "chinatown": (41.8516, -87.6322),
    "little village": (41.8269, -87.7194),
    "bronzeville": (41.8357, -87.6218),
    "rogers park": (42.0084, -87.6685),
    "edgewater": (41.9882, -87.6607),
    "albany park": (41.9693, -87.7193),
    "irving park": (41.9529, -87.7332),
    "avondale": (41.9399, -87.7194),
    "humboldt park": (41.9017, -87.7199),
    "west loop": (41.8831, -87.6458),
    "fulton market": (41.8864, -87.6497),
    "near north side": (41.9010, -87.6340),
    "bucktown": (41.9174, -87.6779),
    "ravenswood": (41.9793, -87.6752),
    "north center": (41.9566, -87.6707),
    "roscoe village": (41.9440, -87.6742),
    "ukrainian village": (41.8941, -87.6793),
    "noble square": (41.8991, -87.6739),
    "near west side": (41.8763, -87.6606),
    "austin": (41.8979, -87.7649),
    "englewood": (41.7791, -87.6466),
    "south shore": (41.7611, -87.5713),
    "woodlawn": (41.7794, -87.5964),
    "beverly": (41.7240, -87.6578),
}

SYSTEM_PROMPT = f"""You are a friendly data analyst assistant that helps people understand their Chicago business survival chances using a statistical model trained on real Chicago business license data.

When a user asks about opening a business or requests a survival prediction:
1. Determine the BUSINESS TYPE to pick the right license (see list below)
2. Determine the LOCATION in Chicago (neighborhood or coordinates)

Supported license types:
{chr(10).join(f"  - {lt}" for lt in LICENSE_TYPES)}

Chicago neighborhoods you know: {", ".join(NEIGHBORHOODS.keys())}

Rules:
- If the user mentions a neighborhood not in your list, pick the geographically nearest one you know.
- If they give a street address, estimate coordinates from the Chicago street grid (State & Madison = 41.8819, -87.6278; each city block ≈ 0.011 degrees).
- Once you have BOTH location and license type, call predict_business_survival immediately — do not ask for confirmation.
- If you only have one piece of information, ask for the other in one friendly sentence.
- Keep all conversational responses under 2 sentences."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_business_survival",
            "description": "Run the Cox Proportional Hazards survival model for a Chicago business. Call this once you know the business location and type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the business location in Chicago (e.g. 41.8827)",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the business location in Chicago (e.g. -87.6233)",
                    },
                    "license_type": {
                        "type": "string",
                        "enum": LICENSE_TYPES,
                        "description": "Business license type",
                    },
                    "zip_code": {
                        "type": "integer",
                        "description": "Chicago ZIP code of the business location (e.g. 60601)",
                    },
                },
                "required": ["latitude", "longitude", "license_type", "zip_code"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Load ACS rent + income lookup (ZIP → (median_rent, median_income))
# ---------------------------------------------------------------------------
def _load_acs() -> dict:
    url = (
        "https://api.census.gov/data/2022/acs/acs5"
        "?get=B25064_001E,B19013_001E"
        "&for=zip%20code%20tabulation%20area:*"
        "&in=state:17"
    )
    try:
        header, *rows = requests.get(url, timeout=20).json()
        lookup = {}
        for row in rows:
            zip_code = int(row[header.index("zip code tabulation area")])
            rent     = float(row[0]) if float(row[0]) > 0 else None
            income   = float(row[1]) if float(row[1]) > 0 else None
            lookup[zip_code] = (rent, income)
        return lookup
    except Exception:
        return {}

ACS_LOOKUP: dict = _load_acs()
_IL_MEDIAN_RENT   = float(pd.Series([v[0] for v in ACS_LOOKUP.values() if v[0]]).median()) if ACS_LOOKUP else 1050.0
_IL_MEDIAN_INCOME = float(pd.Series([v[1] for v in ACS_LOOKUP.values() if v[1]]).median()) if ACS_LOOKUP else 72000.0

# In-memory session store (sufficient for a demo)
sessions: dict = {}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def run_prediction(latitude: float, longitude: float, license_type: str, zip_code: int = 0) -> dict:
    rent, income = ACS_LOOKUP.get(zip_code, (None, None))
    median_rent   = rent   if rent   else _IL_MEDIAN_RENT
    median_income = income if income else _IL_MEDIAN_INCOME

    required_features = list(model.params_.index)
    row = {feat: 0 for feat in required_features}
    row["LATITUDE"]       = latitude
    row["LONGITUDE"]      = longitude
    row["MEDIAN_RENT"]    = median_rent
    row["MEDIAN_INCOME"]  = median_income
    if license_type in required_features:
        row[license_type] = 1

    df = pd.DataFrame([row])
    curve = model.predict_survival_function(df)
    probs = curve.iloc[:, 0].tolist()

    def prob_at(year: float) -> float:
        idx = curve.index.get_indexer([year], method="nearest")[0]
        return round(float(probs[idx]), 4)

    sampled_curve = []
    for t in np.arange(0, 15.5, 0.5):
        idx = curve.index.get_indexer([t], method="nearest")[0]
        sampled_curve.append({"t": round(float(t), 1), "p": round(float(probs[idx]), 4)})

    return {
        "year_1": prob_at(1),
        "year_3": prob_at(3),
        "year_5": prob_at(5),
        "year_10": prob_at(10),
        "curve": sampled_curve,
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    user_message: str = body.get("message", "").strip()
    session_id: str = body.get("session_id") or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    history = sessions[session_id]
    history.append({"role": "user", "content": user_message})

    # First call to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    if response.choices[0].finish_reason == "tool_calls":
        tool_call = msg.tool_calls[0]
        inp = json.loads(tool_call.function.arguments)

        prediction = run_prediction(inp["latitude"], inp["longitude"], inp["license_type"], inp.get("zip_code", 0))

        # Add assistant's tool call to history
        history.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }],
        })

        # Add tool result
        history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(prediction),
        })

        # Get Claude's final narrative response
        final = client.chat.completions.create(
            model="gpt-4o",
            messages=history,
            tools=TOOLS,
            tool_choice="none",
        )

        final_text = final.choices[0].message.content
        history.append({"role": "assistant", "content": final_text})

        return jsonify({
            "type": "prediction",
            "message": final_text,
            "prediction": prediction,
            "meta": {
                "license_type": inp["license_type"],
                "lat": inp["latitude"],
                "lon": inp["longitude"],
            },
            "session_id": session_id,
        })

    # Conversational reply (asking for more info)
    reply = msg.content
    history.append({"role": "assistant", "content": reply})

    return jsonify({
        "type": "question",
        "message": reply,
        "session_id": session_id,
    })


@app.route("/reset", methods=["POST"])
def reset():
    body = request.get_json(force=True)
    sid = body.get("session_id")
    if sid and sid in sessions:
        del sessions[sid]
    return jsonify({"status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
