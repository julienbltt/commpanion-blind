import threading
import llm
from wakeword_detector import WakeWordDetector
import time
import stt
from recorder import AudioRecorder
import classify
from tts import talk_stream

class VoiceAssistant:
    def __init__(self):
        self.glasses = None
        self.is_processing = False
        self.processing_lock = threading.Lock()

        # Configuration of the glasses
        self.VENDOR_ID = 0x17EF
        self.PRODUCT_ID = 0xB813

        # Initialize all modules
        self.recorder = AudioRecorder()
        self.llm = llm.LMStudioResponder(
            model_name="mistralai/mistral-7b-instruct-v0.3",
            system_prompt="Respond with 1 sentence only."
        )
        self.stt_app = stt.SpeechToTextApplication("audio")
        self.classifier = classify.IntentClassifier()

        # Initialize wake word detector
        self.detector = WakeWordDetector(
            wakeword_models=["models/hey_lucy.onnx"]
        )
        self.detector.register_callback("hey_lucy", self.on_wake_word_detected)

        # Auto-select default microphone
        default_mic = self.recorder.mic_selector.get_default_microphone()
        if default_mic:
            self.recorder.set_microphone(default_mic["index"])
            print(f"🎚️ Default microphone selected: {default_mic['name']}")
        else:
            print("❌ No microphones available.")

    def on_voice_trigger(self, event=None):
        """Callback for button or wake word"""
        with self.processing_lock:
            if self.is_processing:
                print("🔄 Already processing. Please wait...")
                return
            self.is_processing = True

        try:
            talk_stream(" What can I do for you?")
            print("🎤 Trigger detected — start listening...")
            self.process_voice_command()
        except Exception as e:
            print(f"❌ Error during voice processing: {e}")
        finally:
            with self.processing_lock:
                self.is_processing = False
            print("✅ Ready for next command!\n")

    def process_voice_command(self):
        """Record, transcribe, and handle LLM response"""
        try:
            time.sleep(0.2)  # Small delay to stabilize

            print("Starting recording...")
            self.recorder.start_recording()
            while self.recorder.is_recording:
                time.sleep(0.1)
            print("Recording finished!")
            self.recorder.save_recording("audio/last_rec.wav")
            self.recorder.cleanup()

            tic = time.time()

            print("Transcribing audio file...")
            prompt = self.stt_app.transcribe()
            print(f"Transcription: {prompt}")

            if not prompt or not prompt.strip():
                print("❌ No speech detected or empty transcription. Try again.")
                return

            if len(prompt.strip()) < 2:
                print("❌ Transcription too short, probably noise. Try again.")
                return

            intent,confidence = self.classifier.classify(prompt)

            if intent == "read_text":
                print("📖 Intent recognized: Reading text...")
            else:
                print(f"🧠 Intent recognized: {intent}")
                response = "For the momnent I can only read text. If you want me to read text, please ask me, aiming the glasses at the text you want me to read."
                talk_stream(response)
                tac = time.time()
                print(f"Total time: {(tac - tic):.2f} seconds")

        except Exception as e:
            print(f"❌ Error in voice processing: {e}")
            import traceback
            traceback.print_exc()

    def on_wake_word_detected(self, wakeword, score):
        print(f"🔊 Wake word '{wakeword}' detected (score: {score:.2f})")
        self.on_voice_trigger()

    def cleanup(self):
        self.detector.stop()
        self.detector.cleanup()
        self.recorder.cleanup()
        self.recorder.mic_selector.cleanup()


    def run(self):
        """Start system with wake word only"""
        print("✅ Voice assistant initialized. Waiting for wake word ('alexa')...")

        try:
            # Start wake word detector in a separate thread to avoid blocking
            self.detector.start()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Exiting...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            self.cleanup()
            self.detector.stop()
            self.detector.cleanup()
            if self.glasses:
                self.glasses.close()


if __name__ == "__main__":
    assistant = VoiceAssistant()

    try:
        assistant.run()

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
