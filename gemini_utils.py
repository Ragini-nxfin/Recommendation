import os
from google import genai
from google.genai import types

def gemini_generate(input_text):
    try:
        print("Input text being sent to Gemini:", input_text)  # Debug print
        
        client = genai.Client(
            api_key="AIzaSyAAHAvhVI4AOSo4FrJcI6pWQQ_GIu23IxA",
        )
        model = "gemini-1.5-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=input_text),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )
        response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response += chunk.text
            
        print("Received response from Gemini:", response)  # Debug print
        return response
    except Exception as e:
        print(f"Error in Gemini API call: {str(e)}")  # Debug print
        return f"Error: {str(e)}"