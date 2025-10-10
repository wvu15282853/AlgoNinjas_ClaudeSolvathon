from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env
print(os.getenv("ANTHROPIC_API_KEY"))

def classify_event_with_claude(event_data):
    """
    Classifies a particle event using the Claude API.

    Args:
        event_data: A dictionary or pandas Series containing the event features
                    ('recoil_energy', 'scintillation_yield', 'ionization_charge').

    Returns:
        A dictionary containing the classification label, confidence score, and
        detailed reasoning, or None if an error occurs.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""
    Analyze the following particle detector event data and classify it as one of the following types: WIMP, Axion-like particle, Sterile neutrino, or Background.
    Provide a confidence score (0-100%) for your classification and detailed reasoning based on the provided features.

    Event Data:
    Recoil Energy: {event_data['recoil_energy']:.4f}
    Scintillation Yield: {event_data['scintillation_yield']:.4f}
    Ionization Charge: {event_data['ionization_charge']:.4f}

    Output Format:
    Classification: [Classification Label]
    Confidence: [Confidence Score]%
    Reasoning: [Detailed reasoning]
    """

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929", # Or another suitable Claude model
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Assuming the response structure contains the classification information
        # You may need to adjust this based on the actual API response format
        response_text = message.content[0].text
        classification = None
        confidence = None
        reasoning = None

        lines = response_text.split('\n')
        for line in lines:
            if line.startswith("Classification:"):
                classification = line.replace("Classification:", "").strip()
            elif line.startswith("Confidence:"):
                confidence_str = line.replace("Confidence:", "").strip().replace("%", "")
                try:
                    confidence = int(confidence_str)
                except ValueError:
                    confidence = None
            elif line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()

        return {
            'classification': classification,
            'confidence': confidence,
            'reasoning': reasoning
        }

    except Exception as e:
        print(f"Error interacting with Claude API: {e}")
        return None

if __name__ == "__main__":
    # Example event data for testing
    example_event = {
        'recoil_energy': 5.1234,
        'scintillation_yield': 0.5678,
        'ionization_charge': 1.2345
    }
    result = classify_event_with_claude(example_event)
    if result:
        print(f"Event Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reasoning: {result['reasoning']}")
# Example usage (requires setting ANTHROPIC_API_KEY environment variable)
# You would typically iterate through your synthetic_df for this
# example_event = synthetic_df.iloc[0]
# classification_result = classify_event_with_claude(example_event)
# if classification_result:
#     print(f"Event Classification: {classification_result['classification']}")
#     print(f"Confidence: {classification_result['confidence']}%")
#     print(f"Reasoning: {classification_result['reasoning']}")
