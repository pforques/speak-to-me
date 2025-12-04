# Voice-to-Codex (PID mode)

Ce projet fait trois choses :
- transcrit ta voix avec Whisper local (`main.py`),
- détecte les consoles ouvertes pour récupérer le PID de Codex (`list_shells.py`),
- envoie un message (texte + Enter) dans la fenêtre Codex ciblée par PID (`send.py`).

## Prérequis
- Windows + Python 3.11+.
- Dépendances installées dans le venv du projet :
  ```powershell
  .venv\Scripts\python -m pip install -r requirements.txt
  ```

## Scripts
- `list_shells.py` : liste les fenêtres console/pseudoconsole visibles (titre, classe, PID, hwnd, cmdline si dispo).
  ```powershell
  .venv\Scripts\python list_shells.py
  ```
- `send.py` : envoie du texte (puis Enter) dans la fenêtre dont le PID est donné.
  ```powershell
  .venv\Scripts\python send.py "Bonjour" --pid 8224
  # --no-submit pour ne pas appuyer sur Enter
  # --delay 0.02 pour régler la pause entre touches
  ```
- `main.py` : écoute micro, transcrit (Whisper local) et envoie chaque transcription via `send.py --pid`.
  ```powershell
  # récupérer d’abord le PID Codex avec list_shells.py
  .venv\Scripts\python main.py --model-path models --codex-pid 8224
  # options utiles :
  # --device cpu|cuda, --compute-type int8|float16, --language fr, --input-device <id>
  # --send-script <chemin\vers\send.py> si tu déplaces le script
  ```

## Modèle Whisper
- Le dossier `models/` contient déjà un modèle local. Référence-le avec `--model-path models`.
- Pas de téléchargement HF nécessaire (network peut rester coupé).
Si tu n’as pas le modèle, télécharge-le depuis la racine du projet :
```powershell
# deps déjà installées via requirements.txt
.venv\Scripts\python -m faster_whisper.download_model --model small --output_dir models
# ou plus léger :
# .venv\Scripts\python -m faster_whisper.download_model --model small-int8 --output_dir models
# si besoin d’auth HF : set HF_TOKEN=... ou huggingface-cli login
```

## Flux typique
1) Tu lances Codex manuellement.
2) Tu listes les consoles pour obtenir son PID :
   ```powershell
   .venv\Scripts\python list_shells.py
   ```
3) Tu tests l’envoi manuel :
   ```powershell
   .venv\Scripts\python send.py "Bonjour" --pid <PID>
   ```
4) Tu démarres l’écoute vocale :
   ```powershell
   .venv\Scripts\python main.py --model-path models --codex-pid <PID>
   ```

## Notes
- `send.py` utilise pywin32 pour taper dans la fenêtre ciblée (fallback WriteConsoleInput quand c’est une console classique).
- Assure-toi que la fenêtre Codex est visible (pas minimisée). Si plusieurs consoles Codex, prends le bon PID dans `list_shells.py`.

## Installation sur une machine neuve (récap)
```powershell
git clone git@github.com:pforques/speak-to-me.git
cd speak-to-me
winget install Python.Python.3.11      # si Python non présent
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
# récupérer/installer le modèle
.venv\Scripts\python -m faster_whisper.download_model --model small --output_dir models
# lancer Codex manuellement, récupérer le PID :
.venv\Scripts\python list_shells.py
# tester l’envoi :
.venv\Scripts\python send.py "Bonjour" --pid <PID>
# lancer le vocal :
.venv\Scripts\python main.py --model-path models --codex-pid <PID>
```
