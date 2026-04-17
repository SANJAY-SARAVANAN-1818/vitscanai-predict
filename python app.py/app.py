from __future__ import annotations

import os
import uuid
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, send_from_directory, url_for, session, redirect
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)
METADATA_FILE = REPORT_DIR / "reports_index.json"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "kn": "ಕನ್ನಡ",
    "hi": "हिन्दी",
    "ta": "தமிழ்",
    "ml": "മലയാളം",
    "te": "తెలుగు",
}

TRANSLATIONS = {
    "en": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI Image Analysis",
        "heading": "Vitscan AI Prediction",
        "lead": "Upload a face or skin-related image and get a heuristic prediction based on color, contrast, and detected facial patterns.",
        "note": "This is an educational demo. It is not a medical diagnosis and should not replace lab tests or a doctor.",
        "choose_image": "Choose Image",
        "analyze_image": "Analyze Image",
        "language": "Language",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "uploaded_image": "Uploaded Image",
        "visual_indicators": "Visual Indicators",
        "recommendations": "Recommendations",
        "medicines": "Medicine Recommendations",
        "extracted_metrics": "Extracted Metrics",
        "download_report": "Download Full Report",
        "report_title": "Detailed Report",
        "report_message": "A full text report has been generated for your analysis.",
        "error_no_image": "Please choose an image to analyze.",
        "error_invalid": "Upload a PNG, JPG, JPEG, or WEBP image.",
        "home": "Home",
        "view_reports": "View Reports",
        "view_uploads": "View Uploads",
        "view_deficiencies": "Vitamin Deficiencies",
        "view_topics": "View Topics",
        "view_medicines": "View Medicine Guidance",
        "patient_name": "Patient Name",
        "patient_age": "Patient Age",
        "patient_gender": "Patient Gender",
        "patient_notes": "Patient Notes",
        "patient_details": "Patient Details",
        "root_cause": "Root Cause",
        "solutions": "Solutions",
        "login": "Login",
        "logout": "Logout",
        "sign_in": "Sign In",
        "username": "Username",
        "password": "Password",
        "admin_portal": "Admin Portal",
        "admin_dashboard": "Admin Dashboard",
    },
    "es": {
        "title": "Vitscan AI",
        "eyebrow": "Análisis de Imágenes Vitscan AI",
        "heading": "Predicción Vitscan AI",
        "lead": "Sube una imagen de rostro o piel y obtén una predicción heurística basada en color, contraste y patrones faciales detectados.",
        "note": "Esto es una demostración educativa. No es un diagnóstico médico y no debe reemplazar análisis de laboratorio ni a un médico.",
        "choose_image": "Seleccionar Imagen",
        "analyze_image": "Analizar Imagen",
        "language": "Idioma",
        "prediction": "Predicción",
        "confidence": "Confianza",
        "uploaded_image": "Imagen Subida",
        "visual_indicators": "Indicadores Visuales",
        "recommendations": "Recomendaciones",
        "medicines": "Recomendaciones de Medicamentos",
        "extracted_metrics": "Métricas Extraídas",
        "download_report": "Descargar Informe Completo",
        "report_title": "Informe Detallado",
        "report_message": "Se ha generado un informe de texto completo para su análisis.",
        "error_no_image": "Por favor elige una imagen para analizar.",
        "error_invalid": "Sube una imagen PNG, JPG, JPEG o WEBP.",
        "home": "Inicio",
        "view_reports": "Ver Informes",
        "view_uploads": "Ver Subidas",
        "patient_name": "Nombre del Paciente",
        "patient_age": "Edad del Paciente",
        "patient_gender": "Género del Paciente",
        "patient_notes": "Notas del Paciente",
        "patient_details": "Detalles del Paciente",
        "root_cause": "Causa Raíz",
        "solutions": "Soluciones",
        "login": "Iniciar Sesión",
        "logout": "Cerrar Sesión",
        "sign_in": "Acceder",
        "username": "Usuario",
        "password": "Contraseña",
        "admin_portal": "Portal de Administrador",
        "admin_dashboard": "Panel de Admin",
    },
    "fr": {
        "title": "Vitscan AI",
        "eyebrow": "Analyse d'Image Vitscan AI",
        "heading": "Prédiction Vitscan AI",
        "lead": "Téléchargez une image de visage ou de peau et obtenez une prédiction heuristique basée sur la couleur, le contraste et les motifs faciaux détectés.",
        "note": "Ceci est une démo pédagogique. Ce n'est pas un diagnostic médical et cela ne doit pas remplacer des analyses de laboratoire ou un médecin.",
        "choose_image": "Choisir une Image",
        "analyze_image": "Analyser l'Image",
        "language": "Langue",
        "prediction": "Prédiction",
        "confidence": "Confiance",
        "uploaded_image": "Image Téléchargée",
        "visual_indicators": "Indicateurs Visuels",
        "recommendations": "Recommandations",
        "medicines": "Recommandations de Médicaments",
        "extracted_metrics": "Métriques Extraites",
        "download_report": "Télécharger le Rapport Complet",
        "report_title": "Rapport Détaillé",
        "report_message": "Un rapport texte complet a été généré pour votre analyse.",
        "error_no_image": "Veuillez choisir une image à analyser.",
        "error_invalid": "Téléchargez une image PNG, JPG, JPEG ou WEBP.",
        "home": "Accueil",
        "view_reports": "Voir les Rapports",
        "view_uploads": "Voir les Téléchargements",
        "patient_name": "Nom du Patient",
        "patient_age": "Âge du Patient",
        "patient_gender": "Sexe du Patient",
        "patient_notes": "Notes du Patient",
        "patient_details": "Informations du Patient",
        "root_cause": "Cause Racine",
        "solutions": "Solutions",
        "login": "Connexion",
        "logout": "Déconnexion",
        "sign_in": "Se Connecter",
        "username": "Nom d'utilisateur",
        "password": "Mot de passe",
        "admin_portal": "Portail Admin",
        "admin_dashboard": "Tableau de Bord Admin",
    },
    "kn": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI ಇಮೇಜ್ ವಿಶ್ಲೇಷಣೆ",
        "heading": "Vitscan AI ಭವಿಷ್ಯ ವಾಣಿ",
        "lead": "ಮುಖ ಅಥವಾ ಚರ್ಮದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಬಣ್ಣ, ಭಿನ್ನತೆ ಮತ್ತು ಮುಖದ ಮಾದರಿಗಳ ಆಧಾರದ ಮೇಲೆ ಅಂದಾಜು ಪಡೆಯಿರಿ.",
        "note": "ಇದು ಶೈಕ್ಷಣಿಕ ಡೆಮೋ ಆಗಿದೆ. ಇದು ವೈದ್ಯಕೀಯ ನಿರ್ಧಾರವಲ್ಲ ಮತ್ತು ಪ್ರಯೋಗಾಲಯದ ಪರೀಕ್ಷೆಗಳನ್ನು ಅಥವಾ ವೈದ್ಯರನ್ನು ಬದಲಾಯಿಸಬಾರದು.",
        "choose_image": "ಚಿತ್ರ ಆಯ್ಕೆಮಾಡಿ",
        "analyze_image": "ವಿಶ್ಲೇಷಿಸಿ",
        "language": "ಭಾಷೆ",
        "prediction": "ಅಂದಾಜು",
        "confidence": "ಭರವಸೆ",
        "uploaded_image": "ಅಪ್‌ಲೋಡ್ ಮಾಡಲಾದ ಚಿತ್ರ",
        "visual_indicators": "ಕಾಣುವ ಸೂಚನೆಗಳು",
        "recommendations": "ಶಿಫಾರಸುಗಳು",
        "extracted_metrics": "ತೀತ ಕണക്കಿತಗಳು",
        "download_report": "पूर्ण ವರದಿ ಡೌನ್ಲೋಡ್ ಮಾಡಿ",
        "report_title": "ವಿಸ್ತೃತ ವರದಿ",
        "report_message": "ನಿಮ್ಮ ವಿಶ್ಲೇಷಣೆಗೆ ಪಠ್ಯ ವರದಿ ಸೃಷ್ಟಿಸಲಾಗಿದೆ.",
        "error_no_image": "ದಯವಿಟ್ಟು ವಿಶ್ಲೇಷಿಸಲು ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.",
        "error_invalid": "PNG, JPG, JPEG ಅಥವಾ WEBP ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ.",
        "home": "ಮೈನ್",
        "view_reports": "ವರದಿಗಳನ್ನು ನೋಡುವುದು",
        "view_uploads": "ಅಪ್‌ಲೋಡ್‌ಗಳನ್ನು ನೋಡಿ",
        "patient_name": "ರೋಗಿಯ ಹೆಸರು",
        "patient_age": "ರೋಗಿಯ ವಯಸ್ಸು",
        "patient_gender": "ಲಿಂಗ",
        "patient_notes": "ರೋಗಿಯ ಟಿಪ್ಪಣಿಗಳು",
        "patient_details": "ರೋಗಿಯ ವಿವರಗಳು",
        "root_cause": "ಮೂಲ ಕಾರಣ",
        "solutions": " ಪರಿಹಾರಗಳು",
        "login": "ಲಾಗಿನ್",
        "logout": "ಲಾಗೌಟ್",
        "sign_in": "ಸೈನ್ ಇನ್",
        "username": "ಬಳಕೆದಾರ ಹೆಸರು",
        "password": "ಗುಪ್ತಪದ",
        "admin_portal": "ಅಡ್ಮಿನ್ ಪೋರ್‍ಟಲ್",
        "admin_dashboard": "ಅಡ್ಮಿನ್ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
    },
    "hi": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI इमेज विश्लेषण",
        "heading": "Vitscan AI भविष्यवाणी",
        "lead": "चेहरे या त्वचा की छवि अपलोड करें और रंग, कंट्रास्ट और पहचाने गए चेहरे के पैटर्न के आधार पर एक अनुमान प्राप्त करें।",
        "note": "यह एक शैक्षिक डेमो है। यह चिकित्सा निदान नहीं है और इसे लैब परीक्षणों या डॉक्टर की जगह नहीं लेना चाहिए।",
        "choose_image": "चित्र चुनें",
        "analyze_image": "विश्लेषण करें",
        "language": "भाषा",
        "prediction": "पूर्वानुमान",
        "confidence": "विश्वास",
        "uploaded_image": "अपलोड की गई छवि",
        "visual_indicators": "दृश्य संकेतक",
        "recommendations": "सिफ़ारिशें",
        "extracted_metrics": "निकाले गए मीट्रिक",
        "download_report": "पूर्ण रिपोर्ट डाउनलोड करें",
        "report_title": "विस्तृत रिपोर्ट",
        "report_message": "आपके विश्लेषण के लिए पूर्ण पाठ रिपोर्ट उत्पन्न की गई है।",
        "error_no_image": "कृपया विश्लेषण के लिए एक छवि चुनें।",
        "error_invalid": "PNG, JPG, JPEG या WEBP छवि अपलोड करें।",
        "home": "होम",
        "view_reports": "रिपोर्ट देखें",
        "view_uploads": "अपलोड देखें",
        "patient_name": "मरीज का नाम",
        "patient_age": "मरीज की आयु",
        "patient_gender": "लिंग",
        "patient_notes": "मरीज के नोट",
        "patient_details": "रोगी विवरण",
        "root_cause": "मूल कारण",
        "solutions": "समाधान",
        "login": "लॉगिन",
        "logout": "लॉगआउट",
        "sign_in": "साइन इन",
        "username": "उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "admin_portal": "एडमिन पोर्टल",
        "admin_dashboard": "एडमिन डैशबोर्ड",
    },
    "ta": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI படம் பகுப்பு",
        "heading": "Vitscan AI கணிப்பு",
        "lead": "முகம் அல்லது தோல் தொடர்புடைய படத்தை பதிவேற்றவும் மற்றும் நிறம், வண்ணமியக்கம் மற்றும் கண்டறியப்பட்ட முக மாதிரிகளின் அடிப்படையில் ஒரு ஊகத்தைப் பெறவும்.",
        "note": "இது ஒரு கல்வி காட்சி. இது மருத்துவ நோயறிதல் அல்ல மற்றும் ஆய்வக சோதனைகளை அல்லது டாக்டரை மாற்றக்கூடாது.",
        "choose_image": "படத்தைத் தேர்ந்தெடு",
        "analyze_image": "பகுப்பாய்வு",
        "language": "மொழி",
        "prediction": "முன்னறிவு",
        "confidence": "நம்பிக்கை",
        "uploaded_image": "பதிவேற்றப்பட்ட படம்",
        "visual_indicators": "காட்சி குறியீடுகள்",
        "recommendations": "பரிந்துரைகள்",
        "extracted_metrics": "பெறப்பட்ட அளவுருக்கள்",
        "download_report": "முழு அறிக்கையை பதிவிறக்குக",
        "report_title": "விரிவான அறிக்கை",
        "report_message": "உங்கள் பகுப்பாய்வுக்காக முழு உரை அறிக்கை உருவாக்கப்பட்டது.",
        "error_no_image": "தயவு செய்து பகுப்பாய்விற்கு ஒரு படத்தைத் தேர்ந்தெடுக்கவும்.",
        "error_invalid": "PNG, JPG, JPEG அல்லது WEBP படத்தை பதிவேற்றவும்.",
        "home": "முகப்பு",
        "view_reports": "அறிக்கைகளைக் காட்டு",
        "view_uploads": "பதிவேற்றங்களைப் பார்க்கவும்",
        "patient_name": "நோயாளர் பெயர்",
        "patient_age": "நோயாளர் வயது",
        "patient_gender": "பாலினம்",
        "patient_notes": "நோயாளர் குறிப்புகள்",
        "patient_details": "நோயாளர் விவரங்கள்",
        "root_cause": "மூல காரணம்",
        "solutions": "தீர்வுகள்",
        "login": "உள்நுழைவு",
        "logout": "வெளியேறு",
        "sign_in": "உள்நுழையவும்",
        "username": "பயனர் பெயர்",
        "password": "கடவுச்சொல்",
        "admin_portal": "அட்மின் பயன்பாட்டுக்கை",
        "admin_dashboard": "அட்மின் போர்ட்டல்",
    },
    "ml": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI ഇമേജ് വിശകലനം",
        "heading": "Vitscan AI പ്രവചന",
        "lead": "മുഖം അല്ലെങ്കിൽ ത്വക്കുമായി ബന്ധപ്പെട്ട ചിത്രം അപ്‌ലോഡ് ചെയ്ത് നിറം, വ്യത്യാസം, കണ്ടെത്തിയ മുഖ മാതൃകകൾ എന്നിവയുടെ അടിസ്ഥാനത്തിൽ ഒരു കണക്കുകൂട്ടൽ ലഭിക്കുക.",
        "note": "ഇത് ഒരു വിദ്യാഭ്യാസ ഡെമോ ആണ്. ഇത് മെഡിക്കൽ രോഗനിർണയം അല്ല, ലാബ് ടെസ്റ്റുകൾ അല്ലെങ്കിൽ ഡോക്ടറെ மாற்றല്ല.",
        "choose_image": "ചിത്രം തിരഞ്ഞെടുക്കുക",
        "analyze_image": "വിശകലനം ചെയ്യുക",
        "language": "ഭാഷ",
        "prediction": "കണക്കാക്കൽ",
        "confidence": "വിശ്വാസം",
        "uploaded_image": "അപ്‌ലോഡ് ചെയ്ത ചിത്രം",
        "visual_indicators": "ദൃശ്യ സൂചികകൾ",
        "recommendations": "സൂചനകൾ",
        "extracted_metrics": "എടുക്കപ്പെട്ട മെട്രിക്കുകൾ",
        "download_report": "പൂർണ്ണ റിപ്പോർട്ട് ഡൗൺലോഡ് ചെയ്യുക",
        "report_title": "വിവരമുള്ള റിപ്പോർട്ട്",
        "report_message": "നിങ്ങളുടെ വിശകലനത്തിനായി പൂർണ്ണ പാഠ റിപ്പോർട്ട് സൃഷ്ടിച്ചു.",
        "error_no_image": "ദയവായി വിശകലനത്തിനായി ഒരു ചിത്രം തിരഞ്ഞെടുക്കുക.",
        "error_invalid": "PNG, JPG, JPEG, അല്ലെങ്കിൽ WEBP ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.",
        "home": "ഹോം",
        "view_reports": "റിപ്പോർട്ടുകൾ കാണുക",
        "view_uploads": "അപ്‌ലോഡുകൾ കാണുക",
        "patient_name": "രോഗിയുടെ പേര്",
        "patient_age": "രോഗിയുടെ പ്രായം",
        "patient_gender": "ലിംഗം",
        "patient_notes": "രോഗിയുടെ കുറിപ്പുകൾ",
        "patient_details": "രോഗിയുടെ വിവരങ്ങൾ",
        "root_cause": "പ്രധാന കാരണം",
        "solutions": "പരിഹാരങ്ങൾ",
        "login": "ലോഗിൻ",
        "logout": "ലോഗ്ഔട്ട്",
        "sign_in": "സൈൻ ഇൻ",
        "username": "ഉപയോക്തൃ പേര്",
        "password": "പാസ്‌വേഡ്",
        "admin_portal": "അഡ്‌മിൻ പോർട്ടൽ",
        "admin_dashboard": "അഡ്‌മിൻ ഡാഷ്‌ബോർഡ്",
    },
    "te": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI చిత్ర విశ్లేషణ",
        "heading": "Vitscan AI భవిష్యవాణి",
        "lead": "ముఖం లేదా చర్మ సంబంధిత చిత్రాన్ని అప్‌లోడ్ చేసి, రంగు, వ్యత్యాసం మరియు గుర్తించిన ముఖ నమూనాల ఆధారంగా ఊహలను పొందండి.",
        "note": "ఇది విద్యా ప్రదర్శన. ఇది వైద్య నిర్ధారణ కాదు మరియు ప్రయోగశాల పరీక్షలు లేదా డాక్టర్‌ను బదులుగా చేసుకోవద్దు.",
        "choose_image": "చిత్రం ఎంచుకోండి",
        "analyze_image": "విశ్లేషించు",
        "language": "భాష",
        "prediction": "భవిష్యవాణి",
        "confidence": "నిర్భయం",
        "uploaded_image": "అప్‌లోడ్ చేసిన చిత్రం",
        "visual_indicators": "దృశ్య సూచికలు",
        "recommendations": "సిఫార్సులు",
        "extracted_metrics": "పొందిన మీట్రిక్స్",
        "download_report": "పూర్తి నివేదికను డౌన్‌లోడ్ చేయండి",
        "report_title": "వివర నివేదిక",
        "report_message": "మీ విశ్లేషణ కోసం పూర్తి పాఠ నివేదిక రూపొందించబడింది.",
        "error_no_image": "దయచేసి విశ్లేషణకు ఒక చిత్రాన్ని ఎంచుకోండి.",
        "error_invalid": "PNG, JPG, JPEG లేదా WEBP చిత్రం‌ను అప్‌లోడ్ చేయండి.",
        "home": "హోమ్",
        "view_reports": "రిపోర్టులను వీక్షించండి",
        "view_uploads": "అప్‌లోడ్స్ ని చూడండి",
        "patient_name": "రోగి పేరు",
        "patient_age": "వయస్సు",
        "patient_gender": "లింగం",
        "patient_notes": "రోగి నోట్స్",
        "patient_details": "రోగి వివరాలు",
        "root_cause": "మూల కారణం",
        "solutions": "పరిష్కారాలు",
        "login": "లాగిన్",
        "logout": "లాగౌట్",
        "sign_in": "సైన్ ఇన్",
        "username": "వినియోగదారు పేరు",
        "password": "రహస్యపదం",
        "admin_portal": "అడ్మిన్ పోర్టల్",
        "admin_dashboard": "అడ్మిన్ డ్యాష్‌బోర్డ్",
    },
}

METRIC_LABELS = {
    "en": {
        "brightness": "Brightness",
        "contrast": "Contrast",
        "red_ratio": "Red Ratio",
        "yellow_ratio": "Yellow Ratio",
        "saturation": "Saturation",
        "pallor_index": "Pallor Index",
    },
    "es": {
        "brightness": "Brillo",
        "contrast": "Contraste",
        "red_ratio": "Proporción Roja",
        "yellow_ratio": "Proporción Amarilla",
        "saturation": "Saturación",
        "pallor_index": "Índice de Palidez",
    },
    "fr": {
        "brightness": "Luminosité",
        "contrast": "Contraste",
        "red_ratio": "Rapport Rouge",
        "yellow_ratio": "Rapport Jaune",
        "saturation": "Saturation",
        "pallor_index": "Indice de Pâleur",
    },
    "kn": {},
    "hi": {},
    "ta": {},
    "ml": {},
    "te": {},
}

LABEL_TRANSLATIONS = {
    "en": {
        "Iron deficiency / anemia": "Iron deficiency / anemia",
        "Vitamin B12 deficiency": "Vitamin B12 deficiency",
        "Vitamin A deficiency": "Vitamin A deficiency",
        "Vitamin C deficiency": "Vitamin C deficiency",
        "No strong visual deficiency signal": "No strong visual deficiency signal",
    },
    "es": {
        "Iron deficiency / anemia": "Deficiencia de hierro / anemia",
        "Vitamin B12 deficiency": "Deficiencia de vitamina B12",
        "Vitamin A deficiency": "Deficiencia de vitamina A",
        "Vitamin C deficiency": "Deficiencia de vitamina C",
        "No strong visual deficiency signal": "Sin señal visual fuerte de deficiencia",
    },
    "fr": {
        "Iron deficiency / anemia": "Carence en fer / anémie",
        "Vitamin B12 deficiency": "Carence en vitamine B12",
        "Vitamin A deficiency": "Carence en vitamine A",
        "Vitamin C deficiency": "Carence en vitamine C",
        "No strong visual deficiency signal": "Pas de signal visuel fort de carence",
    },
}

INDICATOR_TRANSLATIONS = {
    "en": {
        "High facial brightness with reduced red tone suggests pallor.": "High facial brightness with reduced red tone suggests pallor.",
        "Central facial area appears lighter than surrounding region.": "Central facial area appears lighter than surrounding region.",
        "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.": "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.",
        "Lower contrast and lower saturation can reflect dull or dry-looking skin.": "Lower contrast and lower saturation can reflect dull or dry-looking skin.",
        "Yellow-red imbalance may indicate gum or skin irritation patterns.": "Yellow-red imbalance may indicate gum or skin irritation patterns.",
        "The analysis did not find a strong visual abnormality in the extracted image region.": "The analysis did not find a strong visual abnormality in the extracted image region.",
    },
    "es": {
        "High facial brightness with reduced red tone suggests pallor.": "El alto brillo facial con tono rojo reducido sugiere palidez.",
        "Central facial area appears lighter than surrounding region.": "La zona central del rostro parece más clara que la región circundante.",
        "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.": "Un mayor énfasis en el rojo y el contraste puede coincidir con regiones inflamadas de la boca o la lengua.",
        "Lower contrast and lower saturation can reflect dull or dry-looking skin.": "El menor contraste y la menor saturación pueden reflejar piel apagada o seca.",
        "Yellow-red imbalance may indicate gum or skin irritation patterns.": "El desequilibrio amarillo-rojo puede indicar patrones de irritación de encías o piel.",
        "The analysis did not find a strong visual abnormality in the extracted image region.": "El análisis no encontró una anomalía visual fuerte en la región de imagen extraída.",
    },
    "fr": {
        "High facial brightness with reduced red tone suggests pallor.": "Une luminosité faciale élevée avec un ton rouge réduit suggère une pâleur.",
        "Central facial area appears lighter than surrounding region.": "La zone centrale du visage semble plus claire que la région environnante.",
        "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.": "Une plus grande priorité au rouge et au contraste peut correspondre à des zones inflammées de la bouche ou de la langue.",
        "Lower contrast and lower saturation can reflect dull or dry-looking skin.": "Un contraste et une saturation plus faibles peuvent refléter une peau terne ou sèche.",
        "Yellow-red imbalance may indicate gum or skin irritation patterns.": "Un déséquilibre jaune-rouge peut indiquer des motifs d'irritation des gencives ou de la peau.",
        "The analysis did not find a strong visual abnormality in the extracted image region.": "L'analyse n'a pas trouvé d'anomalie visuelle forte dans la région d'image extraite.",
    },
}

SUMMARIES = {
    "en": {
        "Iron deficiency / anemia": "The image shows pallor-like patterns that can be associated with low iron or anemia.",
        "Vitamin B12 deficiency": "The image shows red and inflamed color patterns that may align with Vitamin B12-related signs.",
        "Vitamin A deficiency": "The image looks lower in contrast and saturation, which can match dry or dull visual symptoms.",
        "Vitamin C deficiency": "The color balance suggests irritation-related patterns sometimes seen with low Vitamin C.",
        "No strong visual deficiency signal": "The uploaded image does not show a strong match to the built-in visual deficiency heuristics.",
    },
    "es": {
        "Iron deficiency / anemia": "La imagen muestra patrones similares a la palidez que pueden asociarse con bajo hierro o anemia.",
        "Vitamin B12 deficiency": "La imagen muestra patrones rojos e inflamados que pueden coincidir con signos relacionados con la vitamina B12.",
        "Vitamin A deficiency": "La imagen parece tener menor contraste y saturación, lo que puede coincidir con síntomas visuales secos o apagados.",
        "Vitamin C deficiency": "El equilibrio de color sugiere patrones de irritación a veces vistos con bajo contenido de vitamina C.",
        "No strong visual deficiency signal": "La imagen subida no muestra una coincidencia fuerte con las heurísticas visuales de deficiencia.",
    },
    "fr": {
        "Iron deficiency / anemia": "L'image montre des motifs de pâleur qui peuvent être associés à une carence en fer ou à l'anémie.",
        "Vitamin B12 deficiency": "L'image montre des motifs rouges et enflammés qui peuvent correspondre à des signes liés à la vitamine B12.",
        "Vitamin A deficiency": "L'image semble avoir moins de contraste et de saturation, ce qui peut correspondre à des symptômes visuels secs ou ternes.",
        "Vitamin C deficiency": "L'équilibre des couleurs suggère des motifs d'irritation parfois vus en cas de faible teneur en vitamine C.",
        "No strong visual deficiency signal": "L'image téléchargée ne montre pas de correspondance forte avec les heuristiques visuelles de carence.",
    },
}

ADVICE_TEXTS = {
    "en": {
        "Iron deficiency / anemia": [
            "Consider iron-rich foods such as spinach, beans, red meat, and lentils.",
            "Pair plant-based iron sources with Vitamin C-rich foods to improve absorption.",
            "Seek a blood test before taking supplements.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss B12 testing with a clinician, especially if fatigue or numbness is present.",
            "Common sources include eggs, dairy, fish, and fortified cereals.",
            "Avoid self-diagnosing from images alone.",
        ],
        "Vitamin A deficiency": [
            "Include carrots, sweet potatoes, leafy greens, and eggs in your diet.",
            "If night-vision issues or severe dryness are present, seek medical advice.",
            "A photo cannot confirm deficiency on its own.",
        ],
        "Vitamin C deficiency": [
            "Add citrus fruits, berries, tomatoes, and peppers to meals.",
            "Persistent gum bleeding or bruising should be evaluated clinically.",
            "Lab confirmation is recommended before supplementation.",
        ],
        "No strong visual deficiency signal": [
            "The image does not strongly suggest deficiency, but symptoms still matter.",
            "Maintain a balanced diet with fruits, vegetables, protein, and hydration.",
            "Use lab tests for reliable confirmation.",
        ],
    },
    "es": {
        "Iron deficiency / anemia": [
            "Consume alimentos ricos en hierro como espinacas, frijoles, carne roja y lentejas.",
            "Combina fuentes vegetales de hierro con alimentos ricos en vitamina C para mejorar la absorción.",
            "Busca un análisis de sangre antes de tomar suplementos.",
        ],
        "Vitamin B12 deficiency": [
            "Consulta una prueba de B12 con un clínico, especialmente si hay fatiga o entumecimiento.",
            "Fuentes comunes incluyen huevos, lácteos, pescado y cereales fortificados.",
            "Evita el autodiagnóstico solo con imágenes.",
        ],
        "Vitamin A deficiency": [
            "Incluye zanahorias, batatas, verduras de hoja verde y huevos en tu dieta.",
            "Si hay problemas de visión nocturna o sequedad severa, busca consejo médico.",
            "Una foto no puede confirmar una deficiencia por sí sola.",
        ],
        "Vitamin C deficiency": [
            "Agrega cítricos, bayas, tomates y pimientos a las comidas.",
            "El sangrado de encías persistente o moretones debe evaluarse clínicamente.",
            "Se recomienda confirmación de laboratorio antes de suplementar.",
        ],
        "No strong visual deficiency signal": [
            "La imagen no sugiere fuertemente una deficiencia, pero los síntomas siguen siendo importantes.",
            "Mantén una dieta equilibrada con frutas, verduras, proteínas e hidratación.",
            "Usa pruebas de laboratorio para una confirmación confiable.",
        ],
    },
    "fr": {
        "Iron deficiency / anemia": [
            "Considérez des aliments riches en fer comme les épinards, les haricots, la viande rouge et les lentilles.",
            "Associez les sources végétales de fer à des aliments riches en vitamine C pour améliorer l'absorption.",
            "Faites un test sanguin avant de prendre des suppléments.",
        ],
        "Vitamin B12 deficiency": [
            "Discutez d'un test B12 avec un clinicien, surtout en cas de fatigue ou d'engourdissement.",
            "Les sources courantes incluent les œufs, les produits laitiers, le poisson et les céréales enrichies.",
            "Évitez l'auto-diagnostic uniquement à partir d'images.",
        ],
        "Vitamin A deficiency": [
            "Incluez des carottes, des patates douces, des légumes verts à feuilles et des œufs dans votre alimentation.",
            "Si des problèmes de vision nocturne ou une sécheresse sévère sont présents, consultez un médecin.",
            "Une photo ne peut pas confirmer une carence à elle seule.",
        ],
        "Vitamin C deficiency": [
            "Ajoutez des agrumes, des baies, des tomates et des poivrons aux repas.",
            "Un saignement persistant des gencives ou des ecchymoses doit être évalué cliniquement.",
            "Une confirmation en laboratoire est recommandée avant la supplémentation.",
        ],
        "No strong visual deficiency signal": [
            "L'image ne suggère pas fortement une carence, mais les symptômes restent importants.",
            "Maintenez une alimentation équilibrée avec des fruits, des légumes, des protéines et une hydratation.",
            "Utilisez des tests de laboratoire pour une confirmation fiable.",
        ],
    },
}

MEDICINE_TEXTS = {
    "en": {
        "Iron deficiency / anemia": [
            "Discuss iron supplements such as ferrous sulfate with a clinician.",
            "Consider a multivitamin with iron if recommended by a doctor.",
            "Ask your provider about iron formulations that minimize stomach upset.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss cyanocobalamin or methylcobalamin supplementation with a clinician.",
            "B12 injections may be recommended if absorption is impaired.",
            "Check if a complete B-complex is appropriate for your situation.",
        ],
        "Vitamin A deficiency": [
            "Discuss a vitamin A supplement with your healthcare provider.",
            "Ask if a beta-carotene or retinol formulation is best for you.",
            "Avoid high-dose vitamin A without medical supervision.",
        ],
        "Vitamin C deficiency": [
            "Discuss ascorbic acid supplements with a clinician.",
            "Consider a buffered vitamin C formula if you have stomach sensitivity.",
            "Ask your provider whether a daily 500mg to 1000mg dose is appropriate.",
        ],
        "No strong visual deficiency signal": [
            "No specific supplement is recommended based on the image alone.",
            "Focus on a balanced diet and speak with a clinician before supplementing.",
            "Use lab tests to decide whether any supplements are necessary.",
        ],
    },
    "es": {
        "Iron deficiency / anemia": [
            "Consulte suplementos de hierro como sulfato ferroso con un clínico.",
            "Considere un multivitamínico con hierro si lo recomienda un médico.",
            "Pregunte a su proveedor sobre formulaciones de hierro que minimicen molestias estomacales.",
        ],
        "Vitamin B12 deficiency": [
            "Discuta la suplementación con cianocobalamina o metilcobalamina con un clínico.",
            "Las inyecciones de B12 pueden recomendarse si la absorción es deficiente.",
            "Verifique si un complejo B completo es apropiado para su situación.",
        ],
        "Vitamin A deficiency": [
            "Consulte un suplemento de vitamina A con su proveedor de atención médica.",
            "Pregunte si una formulación de betacaroteno o retinol es mejor para usted.",
            "Evite altas dosis de vitamina A sin supervisión médica.",
        ],
        "Vitamin C deficiency": [
            "Discuta suplementos de ácido ascórbico con un clínico.",
            "Considere una fórmula de vitamina C amortiguada si tiene sensibilidad estomacal.",
            "Pregunte a su proveedor si una dosis diaria de 500 mg a 1000 mg es apropiada.",
        ],
        "No strong visual deficiency signal": [
            "No se recomienda ningún suplemento específico solo con la imagen.",
            "Concéntrese en una dieta equilibrada y hable con un clínico antes de suplementar.",
            "Use pruebas de laboratorio para decidir si necesita suplementos.",
        ],
    },
    "fr": {
        "Iron deficiency / anemia": [
            "Discutez des suppléments de fer comme le sulfate ferreux avec un clinicien.",
            "Envisagez un multivitamine avec du fer si un médecin le recommande.",
            "Demandez à votre prestataire des formulations de fer qui minimisent les maux d'estomac.",
        ],
        "Vitamin B12 deficiency": [
            "Discutez de la supplémentation en cyanocobalamine ou méthylcobalamine avec un clinicien.",
            "Les injections de B12 peuvent être recommandées en cas de mauvaise absorption.",
            "Vérifiez si un complexe de vitamines B complet est approprié pour votre situation.",
        ],
        "Vitamin A deficiency": [
            "Discutez d'un supplément de vitamine A avec votre professionnel de santé.",
            "Demandez si une formulation bêta-carotène ou rétinol est la meilleure pour vous.",
            "Évitez des doses élevées de vitamine A sans surveillance médicale.",
        ],
        "Vitamin C deficiency": [
            "Discutez de suppléments d'acide ascorbique avec un clinicien.",
            "Envisagez une formule de vitamine C tamponnée si vous avez une sensibilité gastrique.",
            "Demandez à votre prestataire si une dose quotidienne de 500 mg à 1000 mg est appropriée.",
        ],
        "No strong visual deficiency signal": [
            "Aucun supplément spécifique n'est recommandé uniquement sur la base de l'image.",
            "Concentrez-vous sur une alimentation équilibrée et parlez à un clinicien avant de prendre des suppléments.",
            "Utilisez des tests de laboratoire pour décider si des suppléments sont nécessaires.",
        ],
    },
}

MEDICINE_IMAGES = {
    "Iron deficiency / anemia": "medicine-images/iron.svg",
    "Vitamin B12 deficiency": "medicine-images/b12.svg",
    "Vitamin A deficiency": "medicine-images/a.svg",
    "Vitamin C deficiency": "medicine-images/c.svg",
    "No strong visual deficiency signal": "medicine-images/none.svg",
}

DEFICIENCY_TOPICS = [
    {
        "slug": "vitamin-a",
        "label": "Vitamin A Deficiency",
        "symptoms": ["Night blindness", "Dry eyes", "Weak immunity"],
        "medicines": ["Vitamin A capsules (Retinol)"],
        "solutions": ["Eat carrots", "Spinach", "Eggs", "Milk"],
    },
    {
        "slug": "vitamin-b1",
        "label": "Vitamin B1 (Thiamine) Deficiency",
        "symptoms": ["Weakness", "Nerve problems (Beriberi)"],
        "medicines": ["Thiamine tablets"],
        "solutions": ["Whole grains", "Nuts", "Seeds"],
    },
    {
        "slug": "vitamin-b2",
        "label": "Vitamin B2 (Riboflavin) Deficiency",
        "symptoms": ["Cracked lips", "Sore throat"],
        "medicines": ["Riboflavin supplements"],
        "solutions": ["Milk", "Eggs", "Green vegetables"],
    },
    {
        "slug": "vitamin-b3",
        "label": "Vitamin B3 (Niacin) Deficiency",
        "symptoms": ["Skin issues", "Diarrhea (Pellagra)"],
        "medicines": ["Niacin tablets"],
        "solutions": ["Meat", "Fish", "Peanuts"],
    },
    {
        "slug": "vitamin-b6",
        "label": "Vitamin B6 Deficiency",
        "symptoms": ["Anemia", "Depression"],
        "medicines": ["Pyridoxine tablets"],
        "solutions": ["Bananas", "Chicken", "Potatoes"],
    },
    {
        "slug": "vitamin-b9",
        "label": "Vitamin B9 (Folic Acid) Deficiency",
        "symptoms": ["Fatigue", "Anemia"],
        "medicines": ["Folic acid tablets"],
        "solutions": ["Leafy greens", "Beans"],
    },
    {
        "slug": "vitamin-b12",
        "label": "Vitamin B12 Deficiency",
        "symptoms": ["Nerve damage", "Weakness"],
        "medicines": ["B12 injections/tablets"],
        "solutions": ["Meat", "Dairy", "Eggs"],
    },
    {
        "slug": "vitamin-c",
        "label": "Vitamin C Deficiency",
        "symptoms": ["Bleeding gums (Scurvy)", "Low immunity"],
        "medicines": ["Vitamin C tablets"],
        "solutions": ["Citrus fruits", "Amla", "Oranges"],
    },
    {
        "slug": "vitamin-d",
        "label": "Vitamin D Deficiency",
        "symptoms": ["Bone pain", "Weakness (Rickets)"],
        "medicines": ["Vitamin D3 supplements"],
        "solutions": ["Sunlight", "Fish", "Fortified milk"],
    },
    {
        "slug": "vitamin-e",
        "label": "Vitamin E Deficiency",
        "symptoms": ["Muscle weakness", "Vision issues"],
        "medicines": ["Vitamin E capsules"],
        "solutions": ["Nuts", "Seeds", "Vegetable oils"],
    },
    {
        "slug": "vitamin-k",
        "label": "Vitamin K Deficiency",
        "symptoms": ["Excess bleeding", "Slow clotting"],
        "medicines": ["Vitamin K injections/tablets"],
        "solutions": ["Green leafy vegetables"],
    },
]

app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=str(BASE_DIR),
    static_url_path="/static",
)
app.secret_key = os.environ.get("SECRET_KEY", "super-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)
EYE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")
)


@dataclass
class PredictionResult:
    label: str
    confidence: int
    summary: str
    indicators: list[str]
    recommendations: list[str]
    medicines: list[str]
    metrics: dict[str, float]
    root_cause: str
    solutions: list[str]
    patient_info: dict[str, str]


def translate_ui(key: str, language: str) -> str:
    return TRANSLATIONS.get(language, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            language = request.args.get("language", "en")
            if language not in SUPPORTED_LANGUAGES:
                language = "en"
            return redirect(url_for("login", language=language, next=request.url))
        return view(*args, **kwargs)
    return wrapped_view


def build_ui(language: str) -> dict[str, str]:
    return {key: translate_ui(key, language) for key in TRANSLATIONS["en"].keys()}


def translate_label(label: str, language: str) -> str:
    return LABEL_TRANSLATIONS.get(language, LABEL_TRANSLATIONS["en"]).get(label, label)


def translate_indicators(indicators: list[str], language: str) -> list[str]:
    translations = INDICATOR_TRANSLATIONS.get(language, INDICATOR_TRANSLATIONS["en"])
    return [translations.get(item, item) for item in indicators]


def translate_summary(label: str, language: str) -> str:
    return SUMMARIES.get(language, SUMMARIES["en"]).get(label, label)


def translate_recommendations(label: str, language: str) -> list[str]:
    return ADVICE_TEXTS.get(language, ADVICE_TEXTS["en"]).get(label, ADVICE_TEXTS["en"].get(label, []))


def translate_medicines(label: str, language: str) -> list[str]:
    return MEDICINE_TEXTS.get(language, MEDICINE_TEXTS["en"]).get(label, MEDICINE_TEXTS["en"].get(label, []))


def build_ui_result(result: PredictionResult, language: str) -> PredictionResult:
    return PredictionResult(
        label=translate_label(result.label, language),
        confidence=result.confidence,
        summary=translate_summary(result.label, language),
        indicators=translate_indicators(result.indicators, language),
        recommendations=translate_recommendations(result.label, language),
        medicines=translate_medicines(result.label, language),
        metrics=result.metrics,
        root_cause=result.root_cause,
        solutions=result.solutions,
        patient_info=result.patient_info,
    )


def build_medicine_items(language: str) -> list[dict[str, object]]:
    labels = [
        "Iron deficiency / anemia",
        "Vitamin B12 deficiency",
        "Vitamin A deficiency",
        "Vitamin C deficiency",
        "No strong visual deficiency signal",
    ]
    items = []
    for label in labels:
        items.append(
            {
                "label": translate_label(label, language),
                "image_url": url_for("static", filename=MEDICINE_IMAGES[label]),
                "summary": translate_summary(label, language),
                "medicines": translate_medicines(label, language),
                "recommendations": translate_recommendations(label, language),
            }
        )
    return items


def load_report_index() -> list[dict]:
    if not METADATA_FILE.exists():
        return []
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_report_index(reports: list[dict]) -> None:
    METADATA_FILE.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


def append_report_record(record: dict) -> None:
    reports = load_report_index()
    reports.append(record)
    save_report_index(reports)


def load_report_record(report_id: str) -> dict | None:
    reports = load_report_index()
    for report in reports:
        if report.get("id") == report_id:
            return report
    return None


def update_report_record(report_id: str, updates: dict) -> None:
    reports = load_report_index()
    for report in reports:
        if report.get("id") == report_id:
            report.update(updates)
            save_report_index(reports)
            return


MEDICINE_PHOTO_DIR = UPLOAD_DIR / "medicine-photos"
MEDICINE_PHOTO_DIR.mkdir(exist_ok=True)

TOPIC_SECTIONS = [
    "prediction",
    "medicines",
    "recommendations",
    "visual_indicators",
    "extracted_metrics",
]

SECTION_LABELS = {
    "prediction": "Prediction",
    "medicines": "Medicine Guidance",
    "recommendations": "Recommendations",
    "visual_indicators": "Visual Indicators",
    "extracted_metrics": "Extracted Metrics",
}

SECTION_DESCRIPTIONS = {
    "prediction": "Review the predicted deficiency, confidence, root cause, and first solutions.",
    "medicines": "Upload a medicine photo for each recommendation and review the suggested supplements.",
    "recommendations": "Review the medical recommendations for the predicted deficiency.",
    "visual_indicators": "Review the visual indicators used by the analysis.",
    "extracted_metrics": "Review the extracted image metrics used to determine the result.",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage) -> Path:
    extension = file_storage.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    target = UPLOAD_DIR / filename
    file_storage.save(target)
    return target


def save_rgb_image(rgb_image: np.ndarray, prefix: str, extension: str = "jpg") -> Path:
    filename = f"{prefix}-{uuid.uuid4().hex}.{extension}"
    target = UPLOAD_DIR / filename
    Image.fromarray(rgb_image.astype(np.uint8)).save(target)
    return target


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def patient_tracking_key(patient_info: dict[str, str]) -> str:
    name = normalize_text(patient_info.get("patient_name", ""))
    age = normalize_text(patient_info.get("patient_age", ""))
    if name:
        return f"{name}|{age}"
    return ""


def derive_age_group(age_text: str) -> str:
    try:
        age = int(age_text)
    except (TypeError, ValueError):
        return "unknown"
    if age < 18:
        return "child"
    if age < 60:
        return "adult"
    return "senior"


def parse_symptoms(symptom_text: str) -> list[str]:
    return [item.strip() for item in symptom_text.replace("\n", ",").split(",") if item.strip()]


def enhance_image_quality(rgb_image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    denoised = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR), None, 3, 3, 7, 21)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    sharpened = cv2.addWeighted(denoised_rgb, 1.18, cv2.GaussianBlur(denoised_rgb, (0, 0), 2.0), -0.18, 0)
    return sharpened, {
        "pipeline": ["clahe_contrast", "denoise", "sharpen"],
        "note": "Auto-enhancement applied before prediction for low-light and low-contrast images.",
    }


def assess_image_quality(rgb_image: np.ndarray) -> dict[str, object]:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < 70 or sharpness < 35:
        quality = "low"
    elif brightness < 110 or sharpness < 70:
        quality = "fair"
    else:
        quality = "good"
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "quality_label": quality,
    }


def build_explainability_overlay(rgb_image: np.ndarray, label: str) -> tuple[np.ndarray, dict[str, object]]:
    overlay = rgb_image.copy()
    height, width = overlay.shape[:2]
    regions = {
        "Iron deficiency / anemia": ("pale eye and mid-face zone", (int(width * 0.22), int(height * 0.16), int(width * 0.56), int(height * 0.28))),
        "Vitamin B12 deficiency": ("lip and tongue zone", (int(width * 0.26), int(height * 0.52), int(width * 0.48), int(height * 0.18))),
        "Vitamin A deficiency": ("eye dryness zone", (int(width * 0.18), int(height * 0.18), int(width * 0.62), int(height * 0.22))),
        "Vitamin C deficiency": ("gum and mouth zone", (int(width * 0.24), int(height * 0.50), int(width * 0.52), int(height * 0.20))),
        "No strong visual deficiency signal": ("overall facial region", (int(width * 0.16), int(height * 0.16), int(width * 0.68), int(height * 0.52))),
    }
    region_name, (x, y, w, h) = regions.get(label, regions["No strong visual deficiency signal"])
    tint = overlay.copy()
    cv2.rectangle(tint, (x, y), (x + w, y + h), (242, 107, 58), thickness=-1)
    cv2.addWeighted(tint, 0.2, overlay, 0.8, 0, overlay)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (242, 107, 58), thickness=4)
    cv2.putText(overlay, region_name, (x, max(24, y - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return overlay, {
        "highlighted_region": region_name,
        "method": "Prototype explainability overlay inspired by Grad-CAM style localization.",
    }


def multimodal_symptom_signal(symptom_text: str, label: str) -> dict[str, object]:
    text = normalize_text(symptom_text)
    keywords = {
        "Iron deficiency / anemia": ["tired", "fatigue", "pale", "weakness", "dizzy"],
        "Vitamin B12 deficiency": ["numb", "tingling", "memory", "weakness", "burning tongue"],
        "Vitamin A deficiency": ["dry eyes", "night vision", "night blindness", "dry skin"],
        "Vitamin C deficiency": ["bleeding gums", "bruising", "gum pain", "slow healing"],
    }
    matches = [word for word in keywords.get(label, []) if word in text]
    return {
        "symptom_input": symptom_text.strip(),
        "matched_keywords": matches,
        "alignment": "strong" if len(matches) >= 2 else "moderate" if matches else "low",
    }


def location_risk_insight(location_text: str) -> dict[str, object]:
    location = normalize_text(location_text)
    if not location:
        return {"location": "", "regional_risk": "No geo-context provided."}
    if any(word in location for word in ["rural", "village", "tribal"]):
        return {"location": location_text, "regional_risk": "Rural profile may increase Vitamin A and iron deficiency screening priority."}
    if any(word in location for word in ["coastal", "urban", "city"]):
        return {"location": location_text, "regional_risk": "Urban or coastal profile may change diet patterns; balanced deficiency screening still advised."}
    return {"location": location_text, "regional_risk": "Location stored for geo-based deficiency monitoring."}


def build_safety_notes(label: str, age_group: str) -> list[str]:
    notes = [
        "Prototype medicine guidance only. Always confirm with a clinician before starting supplements.",
        "Dosage depends on age, medical history, pregnancy status, and lab confirmation.",
    ]
    if age_group == "child":
        notes.append("Children need age-specific dosing and should not follow adult supplement plans.")
    if label == "Vitamin A deficiency":
        notes.append("High-dose Vitamin A can be harmful without medical supervision.")
    if label == "Iron deficiency / anemia":
        notes.append("Iron supplements should ideally follow blood-test confirmation.")
    return notes


def deepfake_screening_report(rgb_image: np.ndarray) -> dict[str, object]:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    left = gray[:, : gray.shape[1] // 2]
    right = np.fliplr(gray[:, gray.shape[1] // 2 : gray.shape[1] // 2 + left.shape[1]])
    symmetry_gap = float(abs(np.mean(left) - np.mean(right)))
    confidence = max(38, min(94, int(92 - symmetry_gap)))
    status = "likely authentic" if confidence >= 65 else "needs review"
    return {
        "status": status,
        "confidence": confidence,
        "note": "Prototype authenticity check based on simple image-consistency heuristics, not a full deepfake detector.",
    }


def early_detection_signal(metrics: dict[str, float]) -> dict[str, object]:
    subtle_score = round(
        max(0.0, (metrics["yellow_ratio"] - 1.0) * 18)
        + max(0.0, (90 - metrics["saturation"]) * 0.12)
        + max(0.0, abs(metrics["red_ratio"] - 1.05) * 40),
        2,
    )
    return {
        "risk_score": subtle_score,
        "interpretation": "subtle change detected" if subtle_score >= 8 else "no strong early signal",
    }


def assess_trend(patient_info: dict[str, str], label: str, confidence: int) -> dict[str, object]:
    tracking_key = patient_tracking_key(patient_info)
    if not tracking_key:
        return {"status": "baseline", "message": "Tracking starts after repeated patient submissions.", "history_count": 0}
    history = [
        item for item in load_report_index()
        if item.get("tracking_key") == tracking_key
    ]
    if not history:
        return {"status": "baseline", "message": "First saved analysis for this patient.", "history_count": 1}
    previous = sorted(history, key=lambda item: item.get("timestamp", ""))[-1]
    previous_label = previous.get("label", "")
    previous_conf = int(previous.get("confidence", 0))
    if previous_label == label and confidence <= previous_conf - 6:
        status = "improving"
    elif previous_label == label and confidence >= previous_conf + 6:
        status = "worsening"
    elif previous_label != label:
        status = "shifted"
    else:
        status = "stable"
    return {
        "status": status,
        "message": f"Compared with the previous record, deficiency status is {status}.",
        "history_count": len(history) + 1,
        "previous_label": previous_label,
        "previous_confidence": previous_conf,
    }


def analytics_snapshot(reports: list[dict]) -> dict[str, object]:
    label_counts = Counter(report.get("label", "Unknown") for report in reports)
    age_groups = Counter(report.get("age_group", "unknown") for report in reports)
    quality_counts = Counter(report.get("quality_report", {}).get("quality_label", "unknown") for report in reports)
    trend_counts = Counter(report.get("trend_report", {}).get("status", "baseline") for report in reports)
    location_counts = Counter()
    group_counts = Counter()
    for report in reports:
        location = report.get("geo_insight", {}).get("location", "")
        if location:
            location_counts[location] += 1
        group_name = report.get("screening_group", "")
        if group_name:
            group_counts[group_name] += 1
    return {
        "total_reports": len(reports),
        "top_labels": label_counts.most_common(5),
        "age_groups": age_groups.most_common(),
        "quality_counts": quality_counts.most_common(),
        "trend_counts": trend_counts.most_common(),
        "locations": location_counts.most_common(5),
        "groups": group_counts.most_common(5),
    }


def patient_timelines(reports: list[dict]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for report in reports:
        key = report.get("tracking_key", "")
        if key:
            grouped[key].append(report)
    timelines = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: item.get("timestamp", ""))
        latest = ordered[-1]
        timelines.append(
            {
                "patient_name": latest.get("patient_info", {}).get("patient_name", "Unknown"),
                "age_group": latest.get("age_group", "unknown"),
                "count": len(ordered),
                "latest_label": latest.get("label", ""),
                "trend_status": latest.get("trend_report", {}).get("status", "baseline"),
                "latest_timestamp": latest.get("timestamp", ""),
            }
        )
    return sorted(timelines, key=lambda item: item["latest_timestamp"], reverse=True)


def local_chatbot_answer(question: str) -> dict[str, object]:
    text = normalize_text(question)
    if not text:
        return {"answer": "Ask about food, symptoms, vitamins, reports, or tracking and I will guide you locally.", "topic": "help"}
    faq = [
        ("vitamin d", "Vitamin D support usually focuses on sunlight exposure, fortified foods, and clinician-guided D3 supplementation."),
        ("iron", "Iron deficiency guidance usually includes iron-rich foods, Vitamin C pairing, and blood-test confirmation before supplements."),
        ("b12", "Vitamin B12 questions usually relate to eggs, dairy, fish, fortified foods, or clinician-guided B12 tablets/injections."),
        ("report", "Reports in this prototype store the prediction, confidence, explanations, safety notes, tracking status, and image outputs."),
        ("tracking", "Tracking compares repeated reports for the same patient and marks the trend as improving, worsening, stable, or shifted."),
        ("quality", "The app enhances low-light images before prediction and records image-quality scores for monitoring."),
    ]
    for keyword, answer in faq:
        if keyword in text:
            return {"answer": answer, "topic": keyword}
    return {
        "answer": "This local assistant supports nutrition screening guidance. Try asking about iron, Vitamin D, reports, tracking, or image quality.",
        "topic": "general",
    }


def generate_report_file(result: PredictionResult, language: str) -> str:
    text_lines = [
        translate_ui("report_title", language),
        "",
        f"{translate_ui('prediction', language)}: {result.label}",
        f"{translate_ui('confidence', language)}: {result.confidence}%",
        "",
        result.summary,
        "",
        f"{translate_ui('patient_details', language)}:",
    ]

    for key, value in result.patient_info.items():
        if value:
            text_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    text_lines.extend(["", f"{translate_ui('root_cause', language)}:", result.root_cause, "", f"{translate_ui('solutions', language)}:"])
    text_lines.extend(f"- {item}" for item in result.solutions)
    text_lines.extend(["", f"{translate_ui('medicines', language)}:"])
    text_lines.extend(f"- {item}" for item in result.medicines)
    text_lines.extend(["", f"{translate_ui('visual_indicators', language)}:"])
    text_lines.extend(f"- {item}" for item in result.indicators)
    text_lines.extend(["", f"{translate_ui('recommendations', language)}:"])
    text_lines.extend(f"- {item}" for item in result.recommendations)
    text_lines.extend(["", f"{translate_ui('extracted_metrics', language)}:"])
    metric_labels = METRIC_LABELS.get(language, METRIC_LABELS["en"])
    text_lines.extend(
        f"- {metric_labels.get(key, key.replace('_', ' ').title())}: {value:.2f}"
        for key, value in result.metrics.items()
    )

    filename = f"report-{uuid.uuid4().hex}.txt"
    target = REPORT_DIR / filename
    target.write_text("\n".join(text_lines), encoding="utf-8")
    return filename


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        return np.array(rgb)


def crop_interest_region(rgb_image: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return rgb_image

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    face = rgb_image[y : y + h, x : x + w]

    eyes = EYE_CASCADE.detectMultiScale(gray[y : y + h, x : x + w], scaleFactor=1.1, minNeighbors=6)
    if len(eyes) > 0:
        return face
    return face


def extract_metrics(rgb_image: np.ndarray) -> dict[str, float]:
    region = crop_interest_region(rgb_image)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    red_channel = region[:, :, 0].astype(np.float32)
    green_channel = region[:, :, 1].astype(np.float32)
    blue_channel = region[:, :, 2].astype(np.float32)

    red_ratio = float(np.mean(red_channel) / (np.mean(gray) + 1e-6))
    yellow_ratio = float((np.mean(red_channel) + np.mean(green_channel)) / (2 * np.mean(blue_channel) + 1e-6))
    saturation = float(np.mean(hsv[:, :, 1]))

    center = region[region.shape[0] // 4 : region.shape[0] * 3 // 4, region.shape[1] // 4 : region.shape[1] * 3 // 4]
    edge = np.concatenate(
        [
            region[: region.shape[0] // 6, :, :].reshape(-1, 3),
            region[-region.shape[0] // 6 :, :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    pallor_index = float(np.mean(center) - np.mean(edge))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "red_ratio": red_ratio,
        "yellow_ratio": yellow_ratio,
        "saturation": saturation,
        "pallor_index": pallor_index,
    }


def classify_deficiency(metrics: dict[str, float]) -> PredictionResult:
    scores = {
        "Iron deficiency / anemia": 0.0,
        "Vitamin B12 deficiency": 0.0,
        "Vitamin A deficiency": 0.0,
        "Vitamin C deficiency": 0.0,
        "No strong visual deficiency signal": 0.0,
    }
    indicators: list[str] = []

    if metrics["brightness"] > 150 and metrics["red_ratio"] < 1.02:
        scores["Iron deficiency / anemia"] += 0.52
        indicators.append("High facial brightness with reduced red tone suggests pallor.")
    if metrics["pallor_index"] > 7:
        scores["Iron deficiency / anemia"] += 0.24
        indicators.append("Central facial area appears lighter than surrounding region.")

    if metrics["red_ratio"] > 1.10 and metrics["contrast"] > 45:
        scores["Vitamin B12 deficiency"] += 0.48
        indicators.append("Higher red emphasis and contrast may align with inflamed mouth or tongue regions.")
    if metrics["saturation"] > 95:
        scores["Vitamin B12 deficiency"] += 0.18

    if metrics["contrast"] < 32 and metrics["saturation"] < 72:
        scores["Vitamin A deficiency"] += 0.45
        indicators.append("Lower contrast and lower saturation can reflect dull or dry-looking skin.")
    if metrics["brightness"] < 105:
        scores["Vitamin A deficiency"] += 0.16

    if metrics["yellow_ratio"] > 1.62 and metrics["contrast"] > 38:
        scores["Vitamin C deficiency"] += 0.41
        indicators.append("Yellow-red imbalance may indicate gum or skin irritation patterns.")
    if 1.02 <= metrics["red_ratio"] <= 1.10 and metrics["saturation"] > 88:
        scores["Vitamin C deficiency"] += 0.17

    scores["No strong visual deficiency signal"] = 0.35
    if metrics["contrast"] > 36 and 1.00 <= metrics["red_ratio"] <= 1.08 and metrics["saturation"] >= 70:
        scores["No strong visual deficiency signal"] += 0.25
    if metrics["brightness"] < 150 and metrics["brightness"] > 95:
        scores["No strong visual deficiency signal"] += 0.15

    label, raw_score = max(scores.items(), key=lambda item: item[1])
    confidence = max(55, min(94, int(raw_score * 100)))

    summaries = {
        "Iron deficiency / anemia": "The image shows pallor-like patterns that can be associated with low iron or anemia.",
        "Vitamin B12 deficiency": "The image shows red and inflamed color patterns that may align with Vitamin B12-related signs.",
        "Vitamin A deficiency": "The image looks lower in contrast and saturation, which can match dry or dull visual symptoms.",
        "Vitamin C deficiency": "The color balance suggests irritation-related patterns sometimes seen with low Vitamin C.",
        "No strong visual deficiency signal": "The uploaded image does not show a strong match to the built-in visual deficiency heuristics.",
    }

    advice = {
        "Iron deficiency / anemia": [
            "Consider iron-rich foods such as spinach, beans, red meat, and lentils.",
            "Pair plant-based iron sources with Vitamin C-rich foods to improve absorption.",
            "Seek a blood test before taking supplements.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss B12 testing with a clinician, especially if fatigue or numbness is present.",
            "Common sources include eggs, dairy, fish, and fortified cereals.",
            "Avoid self-diagnosing from images alone.",
        ],
        "Vitamin A deficiency": [
            "Include carrots, sweet potatoes, leafy greens, and eggs in your diet.",
            "If night-vision issues or severe dryness are present, seek medical advice.",
            "A photo cannot confirm deficiency on its own.",
        ],
        "Vitamin C deficiency": [
            "Add citrus fruits, berries, tomatoes, and peppers to meals.",
            "Persistent gum bleeding or bruising should be evaluated clinically.",
            "Lab confirmation is recommended before supplementation.",
        ],
        "No strong visual deficiency signal": [
            "The image does not strongly suggest deficiency, but symptoms still matter.",
            "Maintain a balanced diet with fruits, vegetables, protein, and hydration.",
            "Use lab tests for reliable confirmation.",
        ],
    }

    medicines = {
        "Iron deficiency / anemia": [
            "Discuss iron supplements such as ferrous sulfate with a clinician.",
            "Consider a multivitamin with iron if recommended by a doctor.",
            "Ask your provider about iron formulations that minimize stomach upset.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss cyanocobalamin or methylcobalamin supplementation with a clinician.",
            "B12 injections may be recommended if absorption is impaired.",
            "Check if a complete B-complex is appropriate for your situation.",
        ],
        "Vitamin A deficiency": [
            "Discuss a vitamin A supplement with your healthcare provider.",
            "Ask if a beta-carotene or retinol formulation is best for you.",
            "Avoid high-dose vitamin A without medical supervision.",
        ],
        "Vitamin C deficiency": [
            "Discuss ascorbic acid supplements with a clinician.",
            "Consider a buffered vitamin C formula if you have stomach sensitivity.",
            "Ask your provider whether a daily 500mg to 1000mg dose is appropriate.",
        ],
        "No strong visual deficiency signal": [
            "No specific supplement is recommended based on the image alone.",
            "Focus on a balanced diet and speak with a clinician before supplementing.",
            "Use lab tests to decide whether any supplements are necessary.",
        ],
    }

    if not indicators:
        indicators.append("The analysis did not find a strong visual abnormality in the extracted image region.")

    root_causes = {
        "Iron deficiency / anemia": "Possible low dietary iron intake or absorption issues.",
        "Vitamin B12 deficiency": "Possible inadequate vitamin B12 intake or absorption.",
        "Vitamin A deficiency": "Possible insufficient vitamin A from diet or poor nutrition.",
        "Vitamin C deficiency": "Possible low vitamin C intake or poor dietary balance.",
        "No strong visual deficiency signal": "No strong visual nutrition deficiency signal detected.",
    }

    solutions = {
        "Iron deficiency / anemia": [
            "Increase iron-rich foods such as spinach, beans, red meat, and lentils.",
            "Combine iron sources with vitamin C-rich foods to improve absorption.",
            "Consult a clinician for blood tests before taking supplements.",
        ],
        "Vitamin B12 deficiency": [
            "Include eggs, dairy, fish, and fortified cereals in the diet.",
            "Monitor neurological symptoms such as tingling or fatigue.",
            "Avoid self-diagnosis from an image alone.",
        ],
        "Vitamin A deficiency": [
            "Add carrots, sweet potatoes, leafy greens, and eggs to meals.",
            "If dry skin or vision changes persist, seek medical advice.",
            "Maintain a balanced diet with colorful vegetables.",
        ],
        "Vitamin C deficiency": [
            "Add citrus fruits, berries, tomatoes, and peppers to your diet.",
            "Watch for persistent gum bleeding or easy bruising.",
            "Use laboratory tests for accurate confirmation.",
        ],
        "No strong visual deficiency signal": [
            "Keep a balanced diet with vegetables, fruits, protein, and hydration.",
            "Monitor symptoms and consult a clinician if concerned.",
            "Use proper tests rather than relying on images alone.",
        ],
    }

    return PredictionResult(
        label=label,
        confidence=confidence,
        summary=summaries[label],
        indicators=indicators,
        recommendations=advice[label],
        medicines=medicines[label],
        metrics=metrics,
        root_cause=root_causes[label],
        solutions=solutions[label],
        patient_info={},
    )


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    language = request.form.get("language", request.args.get("language", "en"))
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    ui = build_ui(language)
    metric_labels = METRIC_LABELS.get(language, METRIC_LABELS["en"])

    if request.method == "POST":
        file = request.files.get("image")
        patient_info = {
            "patient_name": request.form.get("patient_name", "").strip(),
            "patient_age": request.form.get("patient_age", "").strip(),
            "patient_gender": request.form.get("patient_gender", "").strip(),
            "patient_notes": request.form.get("patient_notes", "").strip(),
            "symptom_input": request.form.get("symptom_input", "").strip(),
            "location": request.form.get("location", "").strip(),
            "spoken_symptoms": request.form.get("spoken_symptoms", "").strip(),
            "screening_group": request.form.get("screening_group", "").strip(),
            "capture_frequency": request.form.get("capture_frequency", "").strip(),
            "posture_signal": request.form.get("posture_signal", "").strip(),
        }

        if not file or not file.filename:
            return render_template(
                "index.html",
                ui=ui,
                languages=SUPPORTED_LANGUAGES,
                language=language,
                error=translate_ui("error_no_image", language),
                metric_labels=metric_labels,
            )
        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                ui=ui,
                languages=SUPPORTED_LANGUAGES,
                language=language,
                error=translate_ui("error_invalid", language),
                metric_labels=metric_labels,
            )

        image_path = save_upload(file)
        original_rgb = load_image(image_path)
        enhanced_rgb, enhancement_report = enhance_image_quality(original_rgb)
        enhanced_path = save_rgb_image(enhanced_rgb, "enhanced")
        quality_report = assess_image_quality(original_rgb)
        metrics = extract_metrics(enhanced_rgb)
        prediction = classify_deficiency(metrics)
        prediction.patient_info = patient_info
        localized_result = build_ui_result(prediction, language)
        age_group = derive_age_group(patient_info["patient_age"])
        trend_report = assess_trend(patient_info, localized_result.label, localized_result.confidence)
        symptom_signal = multimodal_symptom_signal(
            ", ".join([patient_info["symptom_input"], patient_info["spoken_symptoms"]]),
            localized_result.label,
        )
        geo_insight = location_risk_insight(patient_info["location"])
        safety_notes = build_safety_notes(localized_result.label, age_group)
        deepfake_report = deepfake_screening_report(original_rgb)
        early_signal = early_detection_signal(localized_result.metrics)
        overlay_rgb, explainability_report = build_explainability_overlay(enhanced_rgb, localized_result.label)
        overlay_path = save_rgb_image(overlay_rgb, "xai")
        report_filename = generate_report_file(localized_result, language)
        image_url = url_for("static_upload", filename=image_path.name)
        enhanced_image_url = url_for("static_upload", filename=enhanced_path.name)
        overlay_image_url = url_for("static_upload", filename=overlay_path.name)
        tracking_key = patient_tracking_key(patient_info)

        append_report_record(
            {
                "id": report_filename,
                "timestamp": datetime.now().isoformat(),
                "language": language,
                "label": localized_result.label,
                "confidence": localized_result.confidence,
                "summary": localized_result.summary,
                "root_cause": localized_result.root_cause,
                "solutions": localized_result.solutions,
                "patient_info": localized_result.patient_info,
                "indicators": localized_result.indicators,
                "medicines": localized_result.medicines,
                "recommendations": localized_result.recommendations,
                "metrics": localized_result.metrics,
                "image_name": image_path.name,
                "enhanced_image_name": enhanced_path.name,
                "overlay_image_name": overlay_path.name,
                "report_name": report_filename,
                "medicine_photos": {},
                "tracking_key": tracking_key,
                "age_group": age_group,
                "quality_report": quality_report,
                "enhancement_report": enhancement_report,
                "trend_report": trend_report,
                "symptom_signal": symptom_signal,
                "geo_insight": geo_insight,
                "safety_notes": safety_notes,
                "deepfake_report": deepfake_report,
                "early_signal": early_signal,
                "explainability_report": explainability_report,
                "screening_group": patient_info["screening_group"],
                "capture_frequency": patient_info["capture_frequency"],
                "posture_signal": patient_info["posture_signal"],
            }
        )

        session["last_report"] = report_filename
        return redirect(
            url_for("topic_page", section="prediction", report_id=report_filename, language=language)
        )

    return render_template(
        "index.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        metric_labels=metric_labels,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    error = None
    next_url = request.args.get("next", "").strip()

    if session.get("logged_in") and request.method == "GET":
        return redirect(next_url or url_for("admin", language=language))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        next_url = request.form.get("next", "").strip()
        if not username or not password:
            error = "Username and password are required"
        elif username == os.environ.get("ADMIN_USER", "admin") and password == os.environ.get("ADMIN_PASS", "admin123"):
            session["logged_in"] = True
            if next_url.startswith("/"):
                return redirect(next_url)
            if next_url.startswith(request.host_url):
                return redirect(next_url)
            return redirect(url_for("admin", language=language))
        else:
            error = "Invalid credentials"

    return render_template(
        "login.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        error=error,
        next_url=next_url,
    )


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))


@app.route("/admin")
@login_required
def admin():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    metric_labels = METRIC_LABELS.get(language, METRIC_LABELS["en"])
    reports = sorted(load_report_index(), key=lambda r: r.get("timestamp", ""), reverse=True)
    uploads = [p.name for p in UPLOAD_DIR.iterdir() if p.is_file()]
    ports = [port.strip() for port in os.environ.get("MULTI_PORTS", "5000,5001,5002").split(",") if port.strip()]
    current_port = os.environ.get("PORT", "5000")

    return render_template(
        "admin.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        reports=reports,
        uploads=uploads,
        ports=ports,
        current_port=current_port,
        metric_labels=metric_labels,
    )


@app.route("/reports")
@login_required
def reports_page():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    reports = sorted(load_report_index(), key=lambda r: r.get("timestamp", ""), reverse=True)
    return render_template(
        "reports.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        reports=reports,
    )


@app.route("/uploaded-images")
@login_required
def uploaded_images():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    images = [p.name for p in UPLOAD_DIR.iterdir() if p.is_file()]
    return render_template(
        "uploads.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        images=images,
    )


@app.route("/medicines")
@login_required
def medicines_page():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    return render_template(
        "medicines.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        medicine_items=build_medicine_items(language),
    )


@app.route("/tracking")
@login_required
def tracking_page():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    reports = sorted(load_report_index(), key=lambda r: r.get("timestamp", ""), reverse=True)
    return render_template(
        "tracking.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        timelines=patient_timelines(reports),
        reports=reports[:12],
    )


@app.route("/analytics")
@login_required
def analytics_page():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    reports = sorted(load_report_index(), key=lambda r: r.get("timestamp", ""), reverse=True)
    return render_template(
        "analytics.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        analytics=analytics_snapshot(reports),
    )


@app.route("/assistant", methods=["GET", "POST"])
@login_required
def assistant_page():
    language = request.args.get("language", "en")
    if request.method == "POST":
        language = request.form.get("language", language)
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    question = ""
    answer = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        answer = local_chatbot_answer(question)
    return render_template(
        "assistant.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        question=question,
        answer=answer,
    )


@app.route("/batch-screening", methods=["GET", "POST"])
@login_required
def batch_screening_page():
    language = request.args.get("language", "en")
    if request.method == "POST":
        language = request.form.get("language", language)
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    batch_results = []
    group_name = ""
    if request.method == "POST":
        files = [item for item in request.files.getlist("images") if item and item.filename]
        group_name = request.form.get("screening_group", "").strip()
        for file in files:
            if not allowed_file(file.filename):
                continue
            image_path = save_upload(file)
            original_rgb = load_image(image_path)
            enhanced_rgb, _ = enhance_image_quality(original_rgb)
            metrics = extract_metrics(enhanced_rgb)
            result = build_ui_result(classify_deficiency(metrics), language)
            batch_results.append(
                {
                    "filename": file.filename,
                    "label": result.label,
                    "confidence": result.confidence,
                    "summary": result.summary,
                    "image_url": url_for("static_upload", filename=image_path.name),
                    "group_name": group_name,
                }
            )
    return render_template(
        "batch.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        batch_results=batch_results,
        group_name=group_name,
    )


@app.route("/topic/<section>")
@login_required
def topic_page(section: str):
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    report_id = request.args.get("report_id") or session.get("last_report")
    report = load_report_record(report_id) if report_id else None
    if not report:
        return redirect(url_for("index", language=language))

    if section not in TOPIC_SECTIONS:
        section = TOPIC_SECTIONS[0]

    ui = build_ui(language)
    section_title = SECTION_LABELS.get(section, section.title())
    section_description = SECTION_DESCRIPTIONS.get(section, "")
    current_index = TOPIC_SECTIONS.index(section)
    prev_url = None
    next_url = None
    if current_index > 0:
        prev_section = TOPIC_SECTIONS[current_index - 1]
        prev_url = url_for("topic_page", section=prev_section, report_id=report_id, language=language)
    if current_index < len(TOPIC_SECTIONS) - 1:
        next_section = TOPIC_SECTIONS[current_index + 1]
        next_url = url_for("topic_page", section=next_section, report_id=report_id, language=language)

    return render_template(
        "topic.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        report=report,
        section=section,
        section_title=section_title,
        section_description=section_description,
        prev_url=prev_url,
        next_url=next_url,
    )


def get_deficiency_topic(slug: str) -> dict | None:
    for topic in DEFICIENCY_TOPICS:
        if topic["slug"] == slug:
            return topic
    return None


MEDICINE_IMAGE_LIBRARY = {
    "vitamin-a": [
        {
            "slug": "vitamin-a-retinol-capsules",
            "title": "Vitamin A capsules",
            "detail": "Reference card for retinol or vitamin A capsule supplements.",
            "accent": "#d68942",
        }
    ],
    "vitamin-b1": [
        {
            "slug": "vitamin-b1-thiamine-tablets",
            "title": "Thiamine tablets",
            "detail": "Reference card for vitamin B1 or thiamine tablet supplements.",
            "accent": "#6f97cc",
        }
    ],
    "vitamin-b2": [
        {
            "slug": "vitamin-b2-riboflavin-supplements",
            "title": "Riboflavin supplements",
            "detail": "Reference image for vitamin B2 riboflavin supplement tablets.",
            "accent": "#c76578",
        }
    ],
    "vitamin-b3": [
        {
            "slug": "vitamin-b3-niacin-tablets",
            "title": "Niacin tablets",
            "detail": "Reference card for vitamin B3 niacin tablet supplements.",
            "accent": "#b57b52",
        }
    ],
    "vitamin-b6": [
        {
            "slug": "vitamin-b6-pyridoxine-tablets",
            "title": "Pyridoxine tablets",
            "detail": "Reference image for vitamin B6 pyridoxine tablet support.",
            "accent": "#d26e60",
        }
    ],
    "vitamin-b9": [
        {
            "slug": "vitamin-b9-folic-acid-tablets",
            "title": "Folic acid tablets",
            "detail": "Reference card for vitamin B9 folic acid supplementation.",
            "accent": "#c56074",
        }
    ],
    "vitamin-b12": [
        {
            "slug": "vitamin-b12-injections-tablets",
            "title": "B12 injections or tablets",
            "detail": "Reference image for vitamin B12 injection or tablet supplements.",
            "accent": "#c84f68",
        }
    ],
    "vitamin-c": [
        {
            "slug": "vitamin-c-tablets",
            "title": "Vitamin C tablets",
            "detail": "Reference card for ascorbic acid or vitamin C tablets.",
            "accent": "#db6c56",
        }
    ],
    "vitamin-d": [
        {
            "slug": "vitamin-d3-supplements",
            "title": "Vitamin D3 supplements",
            "detail": "Reference image for vitamin D3 tablet or capsule supplementation.",
            "accent": "#7c8dac",
        }
    ],
    "vitamin-e": [
        {
            "slug": "vitamin-e-capsules",
            "title": "Vitamin E capsules",
            "detail": "Reference card for vitamin E capsule supplementation.",
            "accent": "#7ea396",
        }
    ],
    "vitamin-k": [
        {
            "slug": "vitamin-k-injections-tablets",
            "title": "Vitamin K injections or tablets",
            "detail": "Reference image for vitamin K injection or tablet support.",
            "accent": "#8a74b0",
        }
    ],
}

MEDICINE_IMAGE_LOOKUP = {
    item["slug"]: item
    for items in MEDICINE_IMAGE_LIBRARY.values()
    for item in items
}


def medicine_images_for_topic(topic: dict) -> list[dict]:
    items = MEDICINE_IMAGE_LIBRARY.get(topic.get("slug", ""), [])
    medicines = topic.get("medicines", [])
    cards = []
    for index, item in enumerate(items):
        card = dict(item)
        card["deficiency"] = topic.get("label", "Vitamin deficiency")
        card["medicine_name"] = medicines[index] if index < len(medicines) else item["title"]
        cards.append(card)
    return cards


def build_medicine_svg(card: dict) -> str:
    medicine_name = card.get("medicine_name", card.get("title", "Medicine"))
    deficiency_name = card.get("deficiency", "Vitamin deficiency")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 720" role="img" aria-labelledby="title desc">
<title id="title">{card["title"]}</title>
<desc id="desc">{card["detail"]}</desc>
<defs>
  <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#fff9f4" />
    <stop offset="100%" stop-color="#f4e5d8" />
  </linearGradient>
  <filter id="shadow">
    <feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="#7f685a" flood-opacity="0.18"/>
  </filter>
</defs>
<rect width="720" height="720" fill="url(#bg)" />
<rect x="42" y="42" width="636" height="636" rx="34" fill="#fffdf9" stroke="#ead9ca" stroke-width="4" filter="url(#shadow)"/>
<circle cx="560" cy="146" r="82" fill="{card["accent"]}" opacity="0.18" />
<circle cx="164" cy="562" r="94" fill="{card["accent"]}" opacity="0.12" />
<text x="88" y="110" font-size="24" font-family="Georgia, serif" fill="#946653">Medicine image</text>
<text x="88" y="158" font-size="40" font-family="Georgia, serif" fill="#2e241e">{medicine_name}</text>
<text x="88" y="198" font-size="26" font-family="Georgia, serif" fill="#6f5b4f">{deficiency_name}</text>
<rect x="112" y="300" width="496" height="278" rx="28" fill="#faf0e6" stroke="{card["accent"]}" stroke-opacity="0.34" stroke-width="4"/>
<rect x="252" y="350" width="216" height="146" rx="30" fill="#ffffff" stroke="#d6c0b0" stroke-width="4"/>
<rect x="284" y="382" width="152" height="82" rx="18" fill="{card["accent"]}" opacity="0.2"/>
<circle cx="312" cy="535" r="34" fill="#ffffff" stroke="#d6c0b0" stroke-width="4"/>
<circle cx="406" cy="535" r="34" fill="#ffffff" stroke="#d6c0b0" stroke-width="4"/>
<circle cx="312" cy="535" r="18" fill="{card["accent"]}" opacity="0.28"/>
<circle cx="406" cy="535" r="18" fill="{card["accent"]}" opacity="0.28"/>
<text x="88" y="626" font-size="24" font-family="Arial, sans-serif" fill="#5b4a42">{card["detail"]}</text>
<text x="88" y="664" font-size="20" font-family="Arial, sans-serif" fill="#8c7769">Educational use only. Consult a clinician before medication.</text>
</svg>"""


SYMPTOM_IMAGE_LIBRARY = {
    "vitamin-a": [
        {
            "slug": "vitamin-a-dry-eyes",
            "title": "Dry irritated eyes",
            "part": "Eyes",
            "detail": "Reference illustration for dryness and irritation around the eyes.",
            "accent": "#d49a49",
        },
        {
            "slug": "vitamin-a-dry-skin",
            "title": "Dry rough skin",
            "part": "Skin",
            "detail": "Reference image showing rough texture and dull skin appearance.",
            "accent": "#c68d42",
        },
    ],
    "vitamin-b1": [
        {
            "slug": "vitamin-b1-foot-tingling",
            "title": "Foot tingling discomfort",
            "part": "Feet",
            "detail": "Reference card for tingling or burning discomfort in the feet.",
            "accent": "#7b9acc",
        },
        {
            "slug": "vitamin-b1-leg-weakness",
            "title": "Leg weakness signs",
            "part": "Legs",
            "detail": "Reference image for weakness and low-energy movement in the legs.",
            "accent": "#6f90c4",
        },
    ],
    "vitamin-b2": [
        {
            "slug": "vitamin-b2-mouth-cracks",
            "title": "Cracks at lip corners",
            "part": "Mouth corners",
            "detail": "Reference illustration for angular cracks around the lips.",
            "accent": "#cc5e74",
        },
        {
            "slug": "vitamin-b2-magenta-tongue",
            "title": "Magenta tongue color",
            "part": "Tongue",
            "detail": "Reference card showing a reddish-purple tongue tone.",
            "accent": "#b84b6d",
        },
    ],
    "vitamin-b3": [
        {
            "slug": "vitamin-b3-neck-rash",
            "title": "Dark neck rash",
            "part": "Neck",
            "detail": "Reference image for dermatitis-like rash around the neck area.",
            "accent": "#8b6b54",
        },
        {
            "slug": "vitamin-b3-hand-rash",
            "title": "Sun-exposed hand rash",
            "part": "Hands",
            "detail": "Reference illustration for rough rash on exposed skin.",
            "accent": "#9b7458",
        },
    ],
    "vitamin-b6": [
        {
            "slug": "vitamin-b6-scaly-lips",
            "title": "Scaly lip irritation",
            "part": "Lips",
            "detail": "Reference image for dry scaling and irritation on the lips.",
            "accent": "#d47467",
        },
        {
            "slug": "vitamin-b6-face-rash",
            "title": "Facial rash patches",
            "part": "Face",
            "detail": "Reference card showing patchy irritated facial skin.",
            "accent": "#c7675f",
        },
    ],
    "vitamin-b9": [
        {
            "slug": "vitamin-b9-pale-conjunctiva",
            "title": "Pale inner eyelid",
            "part": "Eyelids",
            "detail": "Reference illustration for pale conjunctiva linked to folate deficiency symptoms.",
            "accent": "#c79a8e",
        },
        {
            "slug": "vitamin-b9-mouth-ulcers",
            "title": "Mouth ulcer soreness",
            "part": "Mouth",
            "detail": "Reference card showing sore ulcer-like spots inside the mouth.",
            "accent": "#c55d69",
        },
    ],
    "vitamin-b12": [
        {
            "slug": "b12-red-tongue",
            "title": "Red smooth tongue",
            "part": "Tongue",
            "detail": "Reference illustration of glossitis-like redness and smoothness.",
            "accent": "#cf5d68",
        },
        {
            "slug": "b12-mouth-corners",
            "title": "Cracks at mouth corners",
            "part": "Mouth corners",
            "detail": "Reference image showing irritation near the sides of the mouth.",
            "accent": "#c04d62",
        },
    ],
    "vitamin-c": [
        {
            "slug": "vitamin-c-bleeding-gums",
            "title": "Tender bleeding gums",
            "part": "Gums",
            "detail": "Reference illustration showing swollen or bleeding gum symptoms.",
            "accent": "#de6a63",
        },
        {
            "slug": "vitamin-c-bruising-skin",
            "title": "Easy bruising spots",
            "part": "Skin",
            "detail": "Reference image showing bruise-like spots associated with fragile tissues.",
            "accent": "#8e74bc",
        },
    ],
    "vitamin-d": [
        {
            "slug": "vitamin-d-bone-pain-legs",
            "title": "Leg bone pain area",
            "part": "Legs",
            "detail": "Reference image highlighting lower-limb ache and bone discomfort.",
            "accent": "#7d8fb0",
        },
        {
            "slug": "vitamin-d-back-weakness",
            "title": "Back and muscle weakness",
            "part": "Back",
            "detail": "Reference card for posture fatigue and muscle weakness signs.",
            "accent": "#6f87a5",
        },
    ],
    "vitamin-e": [
        {
            "slug": "vitamin-e-hand-numbness",
            "title": "Hand numbness pattern",
            "part": "Hands",
            "detail": "Reference image for numbness or reduced sensation in the hands.",
            "accent": "#82a39a",
        },
        {
            "slug": "vitamin-e-gait-balance",
            "title": "Balance difficulty",
            "part": "Legs",
            "detail": "Reference illustration for unsteady gait or coordination difficulty.",
            "accent": "#6f978b",
        },
    ],
    "vitamin-k": [
        {
            "slug": "vitamin-k-bruised-arms",
            "title": "Arm bruising patches",
            "part": "Arms",
            "detail": "Reference card showing easy bruising over the arms.",
            "accent": "#8b78b2",
        },
        {
            "slug": "vitamin-k-bleeding-gums",
            "title": "Gum bleeding tendency",
            "part": "Gums",
            "detail": "Reference image for repeated minor gum bleeding signs.",
            "accent": "#c9656d",
        },
    ],
}

SYMPTOM_IMAGE_LOOKUP = {
    item["slug"]: item
    for items in SYMPTOM_IMAGE_LIBRARY.values()
    for item in items
}


def symptom_images_for_topic(topic: dict) -> list[dict]:
    items = SYMPTOM_IMAGE_LIBRARY.get(topic.get("slug", ""), [])
    images = []
    for item in items:
        image = dict(item)
        image["deficiency"] = topic.get("label", "Vitamin deficiency")
        images.append(image)
    return images


def build_symptom_svg(image: dict) -> str:
    deficiency_name = image.get("deficiency", "Vitamin deficiency")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 720" role="img" aria-labelledby="title desc">
<title id="title">{image["title"]}</title>
<desc id="desc">{image["detail"]}</desc>
<defs>
  <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#fffaf4" />
    <stop offset="100%" stop-color="#f2e1d2" />
  </linearGradient>
  <filter id="shadow">
    <feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="#7e6658" flood-opacity="0.18"/>
  </filter>
</defs>
<rect width="720" height="720" fill="url(#bg)" />
<rect x="42" y="42" width="636" height="636" rx="34" fill="#fffdf9" stroke="#ead8c7" stroke-width="4" filter="url(#shadow)"/>
<circle cx="570" cy="140" r="86" fill="{image["accent"]}" opacity="0.18" />
<circle cx="160" cy="560" r="96" fill="{image["accent"]}" opacity="0.12" />
<text x="88" y="112" font-size="24" font-family="Georgia, serif" fill="#95634f">Vitamin deficiency symptom image</text>
<text x="88" y="160" font-size="42" font-family="Georgia, serif" fill="#2f251f">{image["title"]}</text>
<text x="88" y="202" font-size="28" font-family="Georgia, serif" fill="#6f5a4f">{deficiency_name}</text>
<rect x="88" y="232" width="180" height="48" rx="24" fill="{image["accent"]}" opacity="0.16"/>
<text x="116" y="264" font-size="24" font-family="Arial, sans-serif" fill="#4c382f">Body part: {image["part"]}</text>
<rect x="116" y="326" width="488" height="258" rx="26" fill="#f9efe5" stroke="{image["accent"]}" stroke-opacity="0.34" stroke-width="4"/>
<ellipse cx="360" cy="454" rx="152" ry="108" fill="#f0c7ae"/>
<ellipse cx="360" cy="454" rx="108" ry="78" fill="{image["accent"]}" opacity="0.28"/>
<ellipse cx="318" cy="428" rx="23" ry="14" fill="#8b5e48" opacity="0.34"/>
<ellipse cx="402" cy="428" rx="23" ry="14" fill="#8b5e48" opacity="0.34"/>
<path d="M302 492 C336 526, 384 526, 418 492" fill="none" stroke="#8e5445" stroke-width="10" stroke-linecap="round"/>
<circle cx="248" cy="388" r="20" fill="{image["accent"]}" opacity="0.18"/>
<circle cx="470" cy="512" r="18" fill="{image["accent"]}" opacity="0.24"/>
<circle cx="224" cy="514" r="15" fill="{image["accent"]}" opacity="0.2"/>
<text x="88" y="626" font-size="24" font-family="Arial, sans-serif" fill="#5b4a42">{image["detail"]}</text>
<text x="88" y="664" font-size="20" font-family="Arial, sans-serif" fill="#8c7769">Educational use only. Not for diagnosis.</text>
</svg>"""


@app.route("/deficiencies")
@login_required
def deficiencies_page():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    return render_template(
        "deficiencies.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        topics=DEFICIENCY_TOPICS,
    )


@app.route("/deficiencies/<slug>")
@login_required
def deficiency_detail(slug: str):
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    topic = get_deficiency_topic(slug)
    if topic is None:
        return redirect(url_for("deficiencies_page", language=language))
    ui = build_ui(language)
    return render_template(
        "deficiency.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        topic=topic,
        medicine_images=medicine_images_for_topic(topic),
        symptom_images=symptom_images_for_topic(topic),
    )


@app.route("/upload-medicine-photo", methods=["POST"])
@login_required
def upload_medicine_photo():
    language = request.form.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    report_id = request.form.get("report_id", "")
    medicine_label = request.form.get("medicine_label", "")
    file = request.files.get("medicine_photo")

    if not report_id or not medicine_label or not file or not file.filename:
        return redirect(url_for("topic_page", section="medicines", report_id=report_id, language=language))
    if not allowed_file(file.filename):
        return redirect(url_for("topic_page", section="medicines", report_id=report_id, language=language))

    extension = file.filename.rsplit(".", 1)[1].lower()
    safe_label = medicine_label.replace(" ", "-").replace("/", "-").lower()
    filename = f"medicine-{report_id[:8]}-{safe_label}-{uuid.uuid4().hex}.{extension}"
    target = MEDICINE_PHOTO_DIR / filename
    file.save(target)

    report = load_report_record(report_id)
    if report is None:
        return redirect(url_for("index", language=language))

    medicine_photos = report.get("medicine_photos", {})
    medicine_photos[medicine_label] = filename
    update_report_record(report_id, {"medicine_photos": medicine_photos})
    session["last_report"] = report_id

    return redirect(url_for("topic_page", section="medicines", report_id=report_id, language=language))


@app.route("/reports/<path:filename>")
@login_required
def download_report(filename: str):
    return send_from_directory(str(REPORT_DIR), filename, as_attachment=True)


@app.route("/uploads/<path:filename>")
@login_required
def static_upload(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/symptom-images/<slug>.svg")
@login_required
def symptom_image(slug: str):
    image = SYMPTOM_IMAGE_LOOKUP.get(slug)
    if image is None:
        return Response("Symptom image not found.", status=404, mimetype="text/plain")
    return Response(build_symptom_svg(image), mimetype="image/svg+xml")


@app.route("/medicine-images/<slug>.svg")
@login_required
def medicine_image(slug: str):
    card = MEDICINE_IMAGE_LOOKUP.get(slug)
    if card is None:
        return Response("Medicine image not found.", status=404, mimetype="text/plain")
    return Response(build_medicine_svg(card), mimetype="image/svg+xml")


@app.route("/debug-vitscan")
@login_required
def debug_vitscan():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    return {
        "base_dir": str(BASE_DIR),
        "template_folder": app.template_folder,
        "static_folder": app.static_folder,
        "ui_title": build_ui(language)["title"],
        "translation_title": TRANSLATIONS[language]["title"],
        "index_exists": (BASE_DIR / "index.html").exists(),
    }


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=debug)
