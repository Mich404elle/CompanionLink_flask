# voicechat_handler.py
from openai import OpenAI
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class VoiceChatHandler:
    def __init__(self):
        self.sample_rate = 44100
        self.channels = 1
        self.dtype = np.int16

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print("Recording...")
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()
        print("Recording complete")
        return recording

    def save_audio(self, recording, filename="temp_recording.wav"):
        """Save recording to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())
        return filename

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper API"""
        try:
            with open(audio_file, 'rb') as af:
                # Using new client format
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=af
                )
            return transcript.text
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def text_to_speech(self, text):
        """Convert text to speech using OpenAI TTS"""
        try:
            print(f"Starting text-to-speech conversion for text: {text[:50]}...")
            
            # Using new client format
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Save to temporary file and convert to base64
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'output.mp3')
            
            # Write the response to file
            response.stream_to_file(temp_path)
            
            # Convert to base64
            with open(temp_path, 'rb') as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
                
            # Clean up
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
            return audio_data
            
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None