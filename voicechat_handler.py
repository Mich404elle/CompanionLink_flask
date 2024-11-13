from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import io
import tempfile

load_dotenv()

class VoiceChatHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def text_to_speech(self, text):
        """
        Convert text to speech using OpenAI's TTS API
        Returns base64 encoded audio data for Melissa's responses
        """
        try:
            print(f"Starting text-to-speech conversion for text: {text[:50]}...")
            
            # Generate speech using OpenAI's TTS
            response = self.client.audio.speech.create(
                model="tts-1",    
                voice="alloy",    
                input=text        
            )
            
            # Convert bytes to base64 string for JSON serialization
            # 1. response.content contains the raw audio bytes
            # 2. b64encode converts these bytes to base64 format
            # 3. decode('utf-8') converts the base64 bytes to a string
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            return audio_base64
            
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio file using OpenAI's Whisper API
        Expects path to an audio file
        """
        try:
            with open(audio_file_path, 'rb') as audio_file:
                # Transcribe using Whisper
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            return transcript.text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

# Test function
def test_voice_handler():
    handler = VoiceChatHandler()
    
    # Test text to speech
    test_text = "Hello, this is a test message!"
    print("Testing text to speech...")
    audio_data = handler.text_to_speech(test_text)
    if audio_data:
        print("✓ Text-to-speech conversion successful")
    else:
        print("⨯ Text-to-speech conversion failed")

if __name__ == "__main__":
    test_voice_handler()