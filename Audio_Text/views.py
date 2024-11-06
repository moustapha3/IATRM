import os
import whisper
from playsound import playsound
from .models import AudioModel
from gtts import gTTS
from PyPDF2 import PdfReader
import docx
from django.shortcuts import render
import google.generativeai as genai
from django.http import JsonResponse, HttpResponse
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from django.shortcuts import render, redirect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

# Téléchargez les ressources nécessaires si ce n'est pas encore fait
nltk.download('punkt')


# Chargement du modèle Whisper une seule fois
model = whisper.load_model("base")  # Changez le modèle selon vos besoins


# Définition de l'Autoencodeur
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def summarize_text_with_autoencoder(text, encoding_dim=64):
    """Résumé de texte avec Autoencodeur pour la réduction de dimension et similarité."""

    # Tokenisation des phrases
    sentences = sent_tokenize(text)
    if len(sentences) == 1:
        return sentences[0]  # Pas besoin de résumé si une seule phrase

    # Conversion des phrases en vecteurs TF-IDF
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences).toarray()
    input_dim = sentence_vectors.shape[1]

    # Initialisation de l'autoencodeur
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

    # Conversion des vecteurs en tenseurs PyTorch
    inputs = torch.Tensor(sentence_vectors)

    # Entraînement de l'autoencodeur
    num_epochs = 100
    for epoch in range(num_epochs):
        encoded, decoded = autoencoder(inputs)
        loss = criterion(decoded, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Extraction des représentations réduites
    with torch.no_grad():
        sentence_embeddings, _ = autoencoder(inputs)

    # Calcul de la similarité par cosinus entre les phrases réduites
    similarity_matrix = cosine_similarity(sentence_embeddings.numpy())

    # Calcul des scores d'attention (moyenne des similarités)
    attention_scores = similarity_matrix.mean(axis=1)
    sentence_scores = {sentences[i]: attention_scores[i] for i in range(len(sentences))}

    # Sélection des phrases avec les scores d'attention les plus élevés
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    summary = " ".join(ranked_sentences)

    return summary

def home(request):
    context = {}
    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            aobj = AudioModel(audioname='temp1', audio=file)
            aobj.save()

            adurl = aobj.audio.path
            transcription = model.transcribe(adurl)
            text = transcription['text']
            lang = transcription['language']


            # Mise à jour du contexte avec la transcription, langue et résumé
            context['text'] = text
            context['lang'] = lang
            try:
                playsound(adurl)
            except:
                try:
                    audio_file = os.path.join("media", "text_audio.mp3")
                    tts = gTTS(text=text, lang=lang, slow=False)
                    tts.save(audio_file)
                    playsound(audio_file)
                except:
                    pass

    return render(request, 'pages/ttshome.html', context)


def translate_text(request):
    context = {}

    # Récupère le texte transcrit et la langue source depuis la session
    text = request.GET.get('text')
    source_lang = request.GET.get('lang')

    # Vérifie si la requête est de type GET et que la langue cible est spécifiée
    if request.method == 'GET' and text and source_lang:
        target_lang = request.GET.get('target_lang')  # Récupère la langue cible

        # Vérifie si la langue cible est spécifiée
        if target_lang:
            try:
                # Effectue la traduction avec la langue cible
                translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
                # Mise à jour du contexte
                context = {
                    'text': text,
                    'translated_text': translated_text,
                    'lang': source_lang,
                    'target_lang': target_lang,
                }
                audio_file = os.path.join("media", "translated_audio.mp3")
                tts = gTTS(text=translated_text, lang=target_lang, slow=False)
                tts.save(audio_file)

                # Lecture du fichier audio
                playsound(audio_file)
            except Exception as e:
                context['error'] = str(e)  # Capture l'erreur pour le contexte
    # Rendu du template avec le contexte mis à jour
    return render(request, 'pages/ttshome.html', context)


def summary_text(request):
    context = {}

    if request.method == 'POST':
        # Récupère 'text', 'translated_text', 'lang', et 'target_lang' du formulaire
        text = request.POST.get('text')
        translated_text = request.POST.get('translated_text')
        lang = request.POST.get('lang')
        target_lang = request.POST.get('target_lang')

        # Résume chaque texte s'ils sont disponibles
        if text:
            summarized_text = summarize_text_with_autoencoder(text)
            context['summarized_text'] = summarized_text
        if translated_text:
            summarized_translated_text = summarize_text_with_autoencoder(translated_text)
            context['summarized_translated_text'] = summarized_translated_text

        # Ajoute les textes et langues au contexte pour affichage dans le template
        context['lang'] = lang
        context['target_lang'] = target_lang

    return render(request, 'pages/ttshome.html', context)


def doc_summary(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')

        if uploaded_file:
            # Extraire le texte du PDF ou du document Word
            if uploaded_file.name.endswith('.pdf'):
                text = ""
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif uploaded_file.name.endswith(('.doc', '.docx')):
                doc = docx.Document(uploaded_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            else:
                return HttpResponse("Format de fichier non supporté.")

            # Résumer le texte extrait
            summarized_doc = summarize_text_with_autoencoder(text)  # Placeholder pour la fonction de résumé
            return render(request, 'pages/ttshome.html', {'summarized_doc': summarized_doc})

    return render(request, 'pages/ttshome.html')
