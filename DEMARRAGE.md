# Démarrage rapide

## Windows
1. Installe Python 3.11 x64.
2. Télécharge ou clone le dépôt.
3. Ouvre PowerShell à la racine et exécute :

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\start.ps1
```

4. Va sur http://localhost:8501.

## Docker Desktop
Dans la racine du projet :

```powershell
docker compose -f compose.yaml up --build
```

Ouvre http://localhost:8501. Pour arrêter : `docker compose down`.

## Premiers pas
Charge la démo dans la barre latérale, puis demande « analyse les données » ou « prévois 30 jours ». Le chat à commandes est local et limité aux fonctions séries temporelles ; aucun LLM généraliste n'est configuré.
