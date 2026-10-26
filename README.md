# TimesFM AI Studio

Interface web sombre inspirée des consoles d'agents IA, dédiée aux séries temporelles : chat à commandes, import CSV, analyse descriptive, prévision, graphique et export.

> Le chat actuel exécute des commandes locales prédéfinies. Ce n'est pas encore un chatbot LLM libre. Le moteur TimesFM 2.5 est tenté en premier ; un fallback statistique est affiché clairement si le chargement du modèle échoue. La dépendance Python est limitée à TimesFM 2.x pour éviter une migration API implicite.

## Fonctions

- Import CSV et choix des colonnes de valeur/date.
- Nettoyage des valeurs manquantes et tri par date.
- Démo synthétique pour tester l'interface sans fichier.
- Statistiques descriptives.
- Prévisions jusqu'à 512 points avec TimesFM 2.5 si disponible.
- Mode statistique de secours avec bande indicative, explicitement identifié.
- Dashboard Plotly, historique de session et téléchargement CSV.
- Démarrage via Windows PowerShell ou Docker Compose.

## Démarrage rapide — Windows

Prérequis : Python 3.11 x64. Dans PowerShell :

```powershell
git clone https://github.com/githubprojetvincent/timesfm-ai-studio.git
cd timesfm-ai-studio
Set-ExecutionPolicy -Scope Process Bypass
.\start.ps1
```

Ouvre ensuite http://localhost:8501.

## Démarrage — Docker Desktop

À la racine du dépôt :

```powershell
docker compose -f compose.yaml up --build
```

Puis http://localhost:8501. Arrêt : `docker compose down`. Les poids téléchargés sont conservés dans le volume `hf-cache`.

Pour changer le port local : définir `TIMESFM_PORT=8502` avant de lancer Compose.

## Utiliser l'agent

1. Clique sur « Charger la démo » ou importe un CSV.
2. Dans le chat, essaie « analyse les données » ou « prévois 30 jours ».
3. Consulte le dashboard puis exporte les prévisions en CSV.

L'import attend au minimum 10 points numériques valides. Si une colonne date est fournie, ses dates invalides sont supprimées et les lignes sont triées chronologiquement.

## Modèle et performances

L'inférence officielle tente de charger `google/timesfm-2.5-200m-pytorch` depuis Hugging Face. Le projet officiel publie également TimesFM 3.0 ; cette application reste volontairement sur le checkpoint 2.5 (poids sous Apache-2.0) et la série de versions Python 2.x tant qu'une migration distincte n'a pas été intégrée et testée. Le premier lancement peut télécharger plusieurs fichiers et demander de la mémoire. Si ce chargement échoue, l'application utilise une extrapolation linéaire de secours et indique ce choix dans le statut du moteur. Ce résultat de secours n'est pas une prédiction TimesFM et les bandes affichées sont indicatives, pas des intervalles de confiance calibrés.

## Développement

```powershell
python -m pip install -r requirements.txt
python -m compileall app.py
```

Tests unitaires disponibles dans `tests/` (ils ne téléchargent pas les poids TimesFM).

## Arborescence

```
app.py             Interface web Streamlit
requirements.txt   Dépendances Python
Dockerfile         Image de conteneur
compose.yaml       Lancement Docker
start.ps1          Installation et lancement Windows
tests/             Tests unitaires
.github/workflows/ Vérification Python automatisée
```

## Sécurité et confidentialité

Les fichiers CSV sont traités dans l'instance locale de l'application et ne sont pas envoyés à un service applicatif dédié par ce code. Le chargement du modèle peut contacter Hugging Face. Ne publie pas de secrets ni de fichiers de données confidentiels dans Git.
