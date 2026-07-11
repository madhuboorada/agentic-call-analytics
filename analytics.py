import os
import json
import time
import tempfile
import glob
from sarvamai import SarvamAI
from google import genai

# 1. Fetch keys directly from environment variables
SARVAM_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not SARVAM_KEY or not GEMINI_KEY:
    raise ValueError("Missing API keys! Run the export commands in your terminal first.")

# 2. Initialize the updated clients
client = SarvamAI(api_subscription_key=SARVAM_KEY)
ai = genai.Client(api_key=GEMINI_KEY)

# 3. Path to your audio file
audio_file_path = "customer_call.wav" 

print("Creating batch transcription job for long audio...")
# Create a batch job using the Saaras v3 model
job = client.speech_to_text_job.create_job(
    model="saaras:v3",
    mode="transcribe",
    language_code="hi-IN",
    with_diarization=True  # Automatically identifies separate speakers!
)

print("Uploading audio file...")
job.upload_files(file_paths=[audio_file_path])

print("Starting batch processing...")
job.start()

print("Waiting for transcription to complete (this might take a minute)...")
job.wait_until_complete()

# Verify if transcription succeeded
file_results = job.get_file_results()
if len(file_results.get("successful", [])) == 0:
    raise Exception("Batch transcription failed.")

# Download the outputs to extract the transcript text
print("Downloading transcript results...")
with tempfile.TemporaryDirectory() as temp_dir:
    job.download_outputs(output_dir=temp_dir)
    
    # Locate the generated JSON transcript file
    json_files = glob.glob(os.path.join(temp_dir, "*.json"))
    if not json_files:
        raise Exception("Transcript data not found in downloaded outputs.")
        
    with open(json_files[0], "r") as f:
        transcript_data = json.load(f)

# Extract full transcript and language code
transcript = transcript_data.get("transcript", "")
lang = transcript_data.get("language_code", "hi-IN")
print(f"Transcription Complete. Detected Language: {lang}")

# 4. Translation Helper (Only running if not already English)
def chunk_text(text, chunk_size=900):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

if lang != "en-IN":
    print("Translating transcript to English...")
    chunks = chunk_text(transcript)
    english_chunks = []
    for i, chunk in enumerate(chunks):
        result = client.text.translate(
            input=chunk,
            source_language_code=lang,
            target_language_code="en-IN"
        )
        english_chunks.append(result.translated_text)
    english_transcript = "\n".join(english_chunks)
else:
    english_transcript = transcript

# 5. Generate Summary using Gemini
qa_prompt = f"You are a professional QA and Call Analytics Assistant. Tasks: 1. Correct grammar 2. Correct spelling 3. Improve readability 4. Create a detailed summary.\n\nTranscript:\n{english_transcript}"

print("Generating summary...")
qa_response = ai.models.generate_content(
    model="gemini-2.5-flash",
    contents=qa_prompt
)
summary = qa_response.text

# 6. Generate Contact Center Insights (Structured JSON)
nba_prompt = f"You are a Contact Center Expert.\n\nTranscript:\n{english_transcript}\n\nSummary:\n{summary}\n\nGenerate a JSON object with: 1. customer_intent 2. sentiment 3. root_cause 4. next_best_actions (list) 5. follow_up_required. Return raw JSON only, no markdown markdown code blocks."

print("Generating structured call insights...")
nba_response = ai.models.generate_content(
    model="gemini-2.5-flash",
    contents=nba_prompt
)

print("\n--- Final Call Analytics Insights (JSON) ---")
print(nba_response.text)