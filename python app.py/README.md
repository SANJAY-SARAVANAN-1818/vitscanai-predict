# Vitscan AI

This project is a simple educational Flask application branded as Vitscan AI that predicts possible vitamin deficiency patterns from an uploaded image using basic image processing heuristics.

## Features

- Upload an image from the browser
- Select a display language (English, Español, Français)
- Detect a face region when possible
- Extract brightness, contrast, color, and saturation metrics
- Predict one of several possible deficiency categories
- Show confidence, indicators, recommendations, and downloadable reports

## Run

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Mobile Adoption

- The app is responsive and designed to work on modern mobile browsers.
- To install on Android, open the app in Chrome and use "Add to Home screen." The app can also be wrapped in a WebView container for native Android packaging.
- To add it on iOS, open the app in Safari and tap "Share > Add to Home Screen." For native iOS deployment, the web app can be embedded into a simple WKWebView wrapper.

## Important Note

This project is a demo for learning image processing and web deployment. It is not a medical tool, and the prediction is based on handcrafted rules rather than a trained clinical model.
