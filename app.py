import random
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import re
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

load_dotenv()

# Ensure ANTHROPIC_API_KEY is set
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

# --- 1️⃣ Define Pydantic schema ---
class ResearchResponse(BaseModel):
    classification: str
    confidence: str
    reasoning: str

# --- 2️⃣ Initialize Claude + parser ---
llm = ChatAnthropic(model="claude-3-haiku-20240307")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_template(
    """Analyze the following particle detector event data and classify it as one of:
    WIMP, Axion-like particle, Sterile neutrino, or Background.

    Event Data:
    Recoil Energy: {recoil_energy} keV
    Scintillation Yield: {scintillation_yield}
    Ionization Charge: {ionization_charge}
    Pulse Shape: {pulse_shape}
    Position (x,y,z): {direction_position}

    Provide:
    - classification label
    - confidence (0-100%)
    - detailed reasoning

    {format_instructions}
    """
).partial(format_instructions=parser.get_format_instructions())

# --- 3️⃣ JSON extraction helper ---
def extract_json_from_response(text: str) -> str:
    """
    Extract JSON object from Claude's response text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return "{}"

app = Flask(__name__)

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_events', methods=['POST'])
def analyze_multiple_events():
    data = request.get_json()
    num_events_str = data.get('num_events', '1')

    try:
        num_events = int(num_events_str)
        if num_events <= 0:
            return jsonify({"error": "Number of events must be positive"}), 400
        # --- REMOVED: if num_events > 10: check ---
    except ValueError:
        return jsonify({"error": "Invalid number of events provided"}), 400

    all_event_results = []

    for i in range(num_events):
        recoil_energy = round(random.uniform(1, 100), 2)
        scintillation_yield = round(random.uniform(10, 200), 2)
        ionization_charge = round(random.uniform(50, 500), 2)
        pulse_shape = round(random.uniform(0.1, 1.0), 2)
        position = [
            round(random.uniform(-5, 5), 2),
            round(random.uniform(-5, 5), 2),
            round(random.uniform(-5, 5), 2),
        ]

        event_data_for_prompt = {
            "recoil_energy": recoil_energy,
            "scintillation_yield": scintillation_yield,
            "ionization_charge": ionization_charge,
            "pulse_shape": pulse_shape,
            "direction_position": position,
        }
        formatted_prompt = prompt.format(**event_data_for_prompt)

        try:
            raw_response = llm.invoke(formatted_prompt)
            content = getattr(raw_response, "content", raw_response)
            if isinstance(content, list):
                content = content[0].get("text", "")

            json_content = extract_json_from_response(content)
            structured = parser.parse(json_content)

            event_result = {
                "id": i + 1,
                "recoil_energy": recoil_energy,
                "scintillation_yield": scintillation_yield,
                "ionization_charge": ionization_charge,
                "pulse_shape": pulse_shape,
                "position": f"({position[0]}, {position[1]}, {position[2]})",
                "classification": structured.classification,
                "confidence": structured.confidence,
                "reasoning": structured.reasoning
            }
            all_event_results.append(event_result)
        except Exception as e:
            print(f"Error processing event {i+1}: {e}")
            all_event_results.append({
                "id": i + 1,
                "recoil_energy": recoil_energy,
                "scintillation_yield": scintillation_yield,
                "ionization_charge": ionization_charge,
                "pulse_shape": pulse_shape,
                "position": f"({position[0]}, {position[1]}, {position[2]})",
                "classification": "Error",
                "confidence": "0%",
                "reasoning": f"Failed to analyze: {str(e)}"
            })

    return jsonify(all_event_results)

if __name__ == '__main__':
    app.run(debug=True)
