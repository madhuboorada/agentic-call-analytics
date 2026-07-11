import os
import json
import tempfile
import glob
import streamlit as st
from sarvamai import SarvamAI
from google import genai

# Set up page configuration
st.set_page_config(
    page_title="Call Analytics Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title
st.title("🎙️ Agentic Call Analytics & Insights")
st.markdown("Upload a customer call audio recording to transcribe, translate, and extract critical operational insights.")

# --- SIDEBAR: BRING YOUR OWN KEY (BYOK) ---
st.sidebar.header("🔑 API Credentials")
st.sidebar.markdown("Provide your individual subscription keys to execute the pipeline.")

# Input fields for API keys (using password type to hide raw strings)
sarvam_api_key = st.sidebar.text_input("Sarvam AI API Key", type="password", help="Input your Sarvam AI platform key.")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Input your Google GenAI platform key.")

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Keys are held in volatile memory for the duration of this execution session and are not stored permanently.")

# --- MAIN INTERFACE: AUDIO UPLOAD ---
st.header("1. Upload Call Recording")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

# Process trigger configuration
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("🚀 Run Call Analytics", type="primary"):
        # Validation checks
        if not sarvam_api_key:
            st.error("Please provide a valid Sarvam AI API Key in the sidebar.")
        elif not gemini_api_key:
            st.error("Please provide a valid Gemini API Key in the sidebar.")
        else:
            try:
                # Execution indicators
                with st.status("Processing analytics pipeline...", expanded=True) as status:
                    
                    # Save the uploaded streaming file to a temporary disk path for processing
                    status.update(label="Staging audio file...", state="running")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_audio:
                        temp_audio.write(uploaded_file.getvalue())
                        temp_audio_path = temp_audio.name

                    # Initialize client environments dynamically with input keys
                    status.update(label="Initializing engine clients...", state="running")
                    client = SarvamAI(api_subscription_key=sarvam_api_key)
                    ai = genai.Client(api_key=gemini_api_key)

                    # Instantiate batch transcription job
                    status.update(label="Creating batch transcription job on Sarvam AI...", state="running")
                    job = client.speech_to_text_job.create_job(
                        model="saaras:v3",
                        mode="transcribe",
                        language_code="hi-IN",
                        with_diarization=True
                    )

                    status.update(label="Uploading audio file payloads...", state="running")
                    job.upload_files(file_paths=[temp_audio_path])

                    status.update(label="Processing audio files via Batch API...", state="running")
                    job.start()
                    job.wait_until_complete()

                    # Extract file metadata results
                    file_results = job.get_file_results()
                    if len(file_results.get("successful", [])) == 0:
                        raise Exception("Batch transcription job failed to complete successfully.")

                    status.update(label="Downloading generated transcript targets...", state="running")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        job.download_outputs(output_dir=temp_dir)
                        json_files = glob.glob(os.path.join(temp_dir, "*.json"))
                        if not json_files:
                            raise Exception("Transcript target payload data not found.")
                            
                        with open(json_files[0], "r") as f:
                            transcript_data = json.load(f)

                    # Extract basic metrics
                    transcript = transcript_data.get("transcript", "")
                    lang = transcript_data.get("language_code", "hi-IN")

                    # Translation workflow if original dialect is non-English
                    if lang != "en-IN":
                        status.update(label=f"Detected system language: '{lang}'. Executing chunked translation...", state="running")
                        chunk_size = 900
                        chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
                        
                        english_chunks = []
                        for idx, chunk in enumerate(chunks):
                            status.update(label=f"Translating segment chunk {idx+1}/{len(chunks)}...", state="running")
                            result = client.text.translate(
                                input=chunk,
                                source_language_code=lang,
                                target_language_code="en-IN"
                            )
                            english_chunks.append(result.translated_text)
                        english_transcript = "\n".join(english_chunks)
                    else:
                        english_transcript = transcript

                    # Summary execution via modern Gemini framework
                    status.update(label="Generating structured analytical summary...", state="running")
                    qa_prompt = f"You are a professional QA and Call Analytics Assistant. Tasks: 1. Correct grammar 2. Correct spelling 3. Improve readability 4. Create a detailed summary.\n\nTranscript:\n{english_transcript}"
                    
                    qa_response = ai.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=qa_prompt
                    )
                    summary = qa_response.text

                    # Strategic Contact Center Next Best Actions extraction
                    status.update(label="Structuring Contact Center insights...", state="running")
                    nba_prompt = f"You are a Contact Center Expert.\n\nTranscript:\n{english_transcript}\n\nSummary:\n{summary}\n\nGenerate a single raw valid JSON object with: 1. customer_intent 2. sentiment 3. root_cause 4. next_best_actions (list) 5. follow_up_required. Return raw JSON text only, without formatting wrappers."
                    
                    nba_response = ai.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=nba_prompt
                    )
                    
                    # Clean the underlying filesystem allocations
                    os.unlink(temp_audio_path)
                    status.update(label="Pipeline processing complete!", state="complete")

                # --- DISPENSING OUTPUT INTERFACES ---
                st.success("Analysis finalized!")
                
                # Layout layout splitting 
                tab1, tab2, tab3 = st.tabs(["📊 Call Insights (JSON)", "📝 Summary", "🔤 Transcribed Text"])
                
                with tab1:
                    st.subheader("Contact Center Parameters")
                    try:
                        # Attempt validation of the JSON content
                        parsed_json = json.loads(nba_response.text.strip())
                        st.json(parsed_json)
                    except Exception:
                        st.text(nba_response.text)
                        
                with tab2:
                    st.subheader("Executive Call Summary")
                    st.write(summary)
                    
                with tab3:
                    st.subheader("English Unified Transcript")
                    st.text_area("Transcript Output", value=english_transcript, height=300)

            except Exception as e:
                st.error(f"Pipeline Interruption: {str(e)}")

else:
    st.info("Awaiting audio upload file execution parameters.")