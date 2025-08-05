# 🧏 EchoSense: Offline Sign Language Interpreter

Translate sign language gestures into spoken or written language — **all on-device** with no internet required.

## 🔧 Setup

```bash
git clone https://github.com/yourusername/echosense.git
cd echosense
pip install -r requirements.txt
streamlit run echosense_app.py
```

## Docker

```bash
docker build -t echosense .
docker run -p 8501:8501 echosense
```

## 🧠 Powered by
- Gemma 3n Vision Model
- Streamlit for UI
- Offline TTS with pyttsx3


---

## ✅ Dataset Tip (For Fine-Tuning)

If you want to fine-tune the model for better sign recognition:
- Use **ASL Alphabet Dataset** (e.g., [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))
- Train a small ResNet18 or MobileNet on ASL → class labels
- Then map predictions to prompts like `"The person is signing: L"` → feed into Gemma3n (optional)

I can help you integrate this if you want a **multi-stage pipeline**.

Let me know if you'd like a GitHub repo template zip or deployment-ready Hugging Face Space version!
