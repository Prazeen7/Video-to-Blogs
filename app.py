from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile, subprocess, os, socket
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Whisper ASR model
whisper_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
whisper_processor = AutoProcessor.from_pretrained(whisper_id)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

# Qwen 2.5 LLM
llm_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_id,
    torch_dtype=torch_dtype,
    device_map="auto"
)

def extract_audio(video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", audio_path
    ], check=True)
    return audio_path

def generate_blog(transcript):
    prompt = (
    "Based strictly on the transcript below, write a human-sounding blog article. "
    "First, write a clear and engaging blog title on its own line starting with 'Title:'.\n"
    "Then leave one empty line, followed by the full blog article body.\n"
    "Avoid using any AI-style formatting such as numbered or bulleted lists, and do not use double dashes (--) or similar punctuation. "
    "Ensure the blog has a clear structure with well-formed paragraphs and a natural, conversational tone. "
    "Do not add external facts or assumptions—only use the transcript content.\n\n"
    
    "Instructions:\n"
    "- Use active voice\n"
    "  - Instead of: 'The meeting was canceled by management.'\n"
    "  - Use: 'Management canceled the meeting.'\n"
    "- Address readers directly with 'you' and 'your'\n"
    "  - Example: 'You'll find these strategies save time.'\n"
    "- Be direct and concise\n"
    "  - Example: 'Call me at 3pm.'\n"
    "- Use simple language\n"
    "  - Example: 'We need to fix this problem.'\n"
    "- Stay away from fluff\n"
    "  - Example: 'The project failed.'\n"
    "- Focus on clarity\n"
    "  - Example: 'Submit your expense report by Friday.'\n"
    "- Vary sentence structures (short, medium, long) to create rhythm\n"
    "  - Example: 'Stop. Think about what happened. Consider how we might prevent similar issues in the future.'\n"
    "- Maintain a natural/conversational tone\n"
    "  - Example: 'But that's not how it works in real life.'\n"
    "- Keep it real\n"
    "  - Example: 'This approach has problems.'\n"
    "- Avoid marketing language\n"
    "  - Avoid: 'Our cutting-edge solution delivers unparalleled results.'\n"
    "  - Use instead: 'Our tool can help you track expenses.'\n"
    "- Simplify grammar\n"
    "  - Example: 'Yeah we can do that tomorrow.'\n"
    "- Avoid AI-philler phrases\n"
    "  - Avoid: 'Let's explore this fascinating opportunity.'\n"
    "  - Use instead: 'Here's what we know.'\n\n"
    
    "Avoid (important!):\n"
    "- Clichés, jargon, hashtags, semicolons, emojis, asterisks, dashes\n"
    "  - Instead of: 'Let's touch base to move the needle on this mission-critical deliverable.'\n"
    "  - Use: 'Let's meet to discuss how to improve this important project.'\n"
    "- Conditional language (could, might, may) when certainty is possible\n"
    "  - Instead of: 'This approach might improve results.'\n"
    "  - Use: 'This approach improves results.'\n"
    "- Redundancy and repetition (remove fluff!)\n\n"
    
    f"Transcript:\n{transcript}"
)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([chat], return_tensors="pt").to(llm_model.device)
    output = llm_model.generate(input_ids.input_ids, max_new_tokens=2048, temperature=0.8)
    generated = output[0][len(input_ids.input_ids[0]):]
    output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Split title and body if possible
    if output_text.lower().startswith("title:"):
        parts = output_text.split("\n", 1)
        title = parts[0][6:].strip()  # after 'Title:'
        body = parts[1].strip() if len(parts) > 1 else ""
    else:
        title = ""
        body = output_text

    return title, body

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    audio_path = extract_audio(video_path)
    result = asr_pipeline(audio_path)
    transcript = result["text"]
    os.remove(audio_path)

    title, blog_body = generate_blog(transcript)

    # Return combined blog text with title on top
    combined_blog = f"Title: {title}\n\n{blog_body}" if title else blog_body

    return jsonify({
        "filename": filename,
        "transcript": transcript,
        "blog": combined_blog
    })

if __name__ == "__main__":
    ip = get_local_ip()
    print(f"🚀 App running on: http://{ip}:4500")
    app.run(host="0.0.0.0", port=4500)
