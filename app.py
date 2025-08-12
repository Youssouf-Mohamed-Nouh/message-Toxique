import pandas as pd
import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    initial_sidebar_state='expanded',
    page_title='Prédicteur Message Toxique - Youssouf',
    page_icon='🗣️',
    layout='wide'
)

@st.cache_resource
def charger_model():
    try:
        model = joblib.load('naivebayestoxique.pkl')  # Nom du modèle toxique
        features = model.named_steps['tfidf'].get_feature_names_out()
        return model, features
    except FileNotFoundError as e:
        st.error(f'Erreur : fichier manquant - {e}')
        st.stop()
    except Exception as e:
        st.error(f'Erreur lors du chargement : {e}')
        st.stop()

model, features = charger_model()

# --- HEADER ---
st.markdown('''
<style>
.main-header{
   background: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 100%);
   padding:2.2rem;
   border-radius:50px;
   margin-bottom:2rem;
   text-align:center;
   box-shadow: 0 20px 50px rgba(0,0,0,0.1);
}
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class='main-header'>
<h1>🗣️ Prédicteur Message Toxique Twitter</h1>
<p style='font-size:20px;'>Développé par - <strong>Youssouf</strong> Assistant Intelligent</p>
</div>
''', unsafe_allow_html=True)

# --- SIDEBAR ---
st.markdown('''
<style>
.friendly-info {
    background: #ffe3e3;
    padding: 2rem;
    border-radius: 15px;
    border-left: 5px solid #ff6b6b;
    margin: 1.5rem 0;
}
.encouragement {
    background: linear-gradient(135deg, #fff0f0, #ffe5e5);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border-left: 5px solid #ff3b3b;
}
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🤖 À propos de votre assistant")
    st.markdown("""
    <div class="friendly-info">
        <h4>Comment je fonctionne ?</h4>
        <p>• J'utilise un modèle Naive Bayes entraîné sur des milliers de messages Twitter</p>
        <p>• Ma précision est d'environ 90%</p>
        <p>• Je respecte votre vie privée</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💡 Rappel important")
    st.markdown("""
    <div class="encouragement">
        <p><strong>Gardez en tête :</strong></p>
        <p>✨ Je suis un outil d'aide, pas un filtre parfait contre la toxicité</p>
    </div>
    """, unsafe_allow_html=True)

# --- FORMULAIRE ---
st.markdown('''
<h2 style='color:#343a40;text-align:center;margin-bottom:25px'> 📝 Texte du tweet à analyser</h2>
''' , unsafe_allow_html=True)

with st.form(key='formulaire_tweet'):
    tweet_text = st.text_area("Collez votre tweet ici :", height=200)
    submitted = st.form_submit_button("Analyser")

# --- PREDICTION ---
if submitted:
    if not tweet_text.strip():
        st.warning("Merci de saisir un texte de tweet valide.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                prediction = model.predict([tweet_text])[0]
                proba = model.predict_proba([tweet_text])[0]
                confiance = proba[prediction] * 100
                
                if prediction == 1:
                    label = 'Toxique'
                    couleur = '#d62828'  # rouge profond
                    emoji = '🛑'
                    commentaire = (
                        "Ce message semble contenir des propos toxiques ou offensants. "
                        "Il est conseillé d'éviter de le partager tel quel."
                    )
                else:
                    label = 'Non toxique'
                    couleur = '#2a9d8f'  # vert doux
                    emoji = '✅'
                    commentaire = (
                        "Ce message paraît respectueux et ne contient pas de langage toxique détecté."
                    )
                
                # Affichage résultat avec pastille et emoji
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:center; gap:1rem; margin-bottom:0.5rem;">
                    <div style="width:28px; height:28px; background-color:{couleur}; border-radius:50%;"></div>
                    <h3 style="color:{couleur}; margin:0;">{emoji}  ==> <strong>{label}</strong></h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Affichage du commentaire explicatif
                st.info(commentaire)
                
                # Affichage confiance en caption
                st.caption(f"Confiance du modèle : {confiance:.1f}%")
                
                # Conseil si confiance faible
                if confiance < 75:
                    st.warning("ℹ️ La confiance du modèle est un peu faible, prenez ce résultat avec prudence.")
                    
            except Exception as e:
                st.error(f"⚠️ Une erreur est survenue lors de la prédiction : {e}")



# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🗣️ Votre Assistant Message Toxique Twitter</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous aider à détecter la toxicité sur Twitter
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2025 - Mis à jour régulièrement pour améliorer la précision
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil complète mais ne remplace jamais une analyse humaine approfondie
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
