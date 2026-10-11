import os
import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xlm_roberta_fake_news")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()


def predict_news(title, article_text, language):
    title = (title or "").strip()
    article_text = (article_text or "").strip()

    if not title and not article_text:
        return (
            "Please enter a headline or article text.",
            {"FAKE": 0.0, "REAL": 0.0},
            "No analysis performed.",
            "No verified supporting evidence is available."
        )

    combined_text = f"{title} {article_text}".strip()

    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    p_fake = float(probs[0].item())
    p_real = float(probs[1].item())

    predicted_index = int(torch.argmax(probs).item())
    predicted_label = model.config.id2label[predicted_index]
    confidence = max(p_fake, p_real) * 100

    prediction = (
        f"## Prediction: {predicted_label}\n\n"
        f"**Model confidence: {confidence:.4f}%**\n\n"
        f"**Selected language: {language}**"
    )

    probabilities = {
        "REAL": p_real * 100,
        "FAKE": p_fake * 100
    }

    confidence_note = (
        "Probabilities are model estimates, not proof of factual truth. "
        "The model can make mistakes, including on unfamiliar topics."
    )

    evidence_note = (
        "## Evidence status\n\n"
        "**No verified supporting evidence is available.** The current "
        "retrieval candidates have not been established as relevant "
        "evidence for the article's specific claims. Similarity scores "
        "alone cannot verify a claim."
    )

    return prediction, probabilities, confidence_note, evidence_note


with gr.Blocks(title="Multilingual Fake News Detection") as demo:
    gr.Markdown(
        "# Multilingual Fake News Detection\n"
        "### XLM-RoBERTa Research Prototype\n"
        "Analyze news in English, Hindi, or Gujarati. Predictions do not "
        "independently establish factual accuracy."
    )

    with gr.Row():
        with gr.Column():
            language = gr.Dropdown(
                ["English", "Hindi", "Gujarati"],
                value="English",
                label="News language"
            )
            title = gr.Textbox(label="News headline", lines=2)
            article = gr.Textbox(label="News article text", lines=8)
            analyze = gr.Button("Analyze News", variant="primary")
            clear = gr.ClearButton(
                [language, title, article],
                value="Clear"
            )

        with gr.Column():
            prediction = gr.Markdown()
            probabilities = gr.Label(
                label="Class probabilities (%)",
                num_top_classes=2
            )
            confidence_note = gr.Markdown()

    evidence = gr.Markdown()

    analyze.click(
        predict_news,
        inputs=[title, article, language],
        outputs=[prediction, probabilities, confidence_note, evidence]
    )

if __name__ == "__main__":
    demo.launch()
