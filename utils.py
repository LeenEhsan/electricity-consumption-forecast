import json
import csv
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from PyPDF2 import PdfReader

def extract_text(path, ext):
    ext = ext.lower()

    try:
        if ext == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data)

        elif ext == "csv":
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                return "\n".join([", ".join(row) for row in reader])

        elif ext == "pdf":
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text if len(text.strip()) >= 10 else ""

        elif ext in ["png", "jpg", "jpeg"]:
            image = Image.open(path).convert("L")
            image = image.filter(ImageFilter.SHARPEN)
            image = ImageEnhance.Contrast(image).enhance(2.0)
            text = pytesseract.image_to_string(image)
            print(f"[OCR] Extracted from image: {text[:200]}")  # DEBUG
            return text if len(text.strip()) >= 10 else ""

        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    except Exception as e:
        print(f"[extract_text] Error extracting from {path}: {e}")
        return ""


def classify_report_text(text, filename=None):
    from app import classifier_model, vectorizer

    if classifier_model is None or vectorizer is None:
        return "unknown", ["Classifier not initialized"]

    # Fallback to filename if text is empty or too short
    if not text or len(text.strip()) < 10:
        print("[AI] Empty or short text, using filename fallback.")
        text = (filename or "unknown").lower()

    try:
        X = vectorizer.transform([text])
        predicted_category = classifier_model.predict(X)[0]

        # Explanation logic
        feature_names = vectorizer.get_feature_names_out()
        class_index = list(classifier_model.classes_).index(predicted_category)
        class_weights = classifier_model.coef_[class_index]
        nonzero_indices = X.nonzero()[1]

        contributions = {
            feature_names[i]: float(class_weights[i]) for i in nonzero_indices
        }

        top_keywords = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        explanation = [kw for kw, _ in top_keywords[:5]]

        return predicted_category, explanation

    except Exception as e:
        return "other", [f"Explanation failed: {str(e)}"]

