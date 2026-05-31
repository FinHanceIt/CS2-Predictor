# Cum pui CS2-Predictor online (Streamlit Community Cloud)

Aplicația e Streamlit (server Python), deci **nu** merge pe Netlify. Casa potrivită e
**Streamlit Community Cloud** — e gratuit, rulează tot (butoanele, modelele, stratul
Claude) și se actualizează singur de fiecare dată când împingi cod nou pe GitHub.

Sunt trei pași. Eu am pregătit deja codul (config, `.gitignore` cu secretele excluse,
un instantaneu de date ca aplicația să nu fie goală la prima deschidere). Pașii care
cer **contul tău** îi faci tu — eu nu pot să mă autentific în GitHub în locul tău.

---

## Pasul 1 — Urcă codul pe GitHub

### Varianta A — GitHub Desktop (recomandat, fără comenzi)

1. Instalează **GitHub Desktop** (desktop.github.com) și conectează-te cu contul tău.
2. `File → Add local repository` și alege folderul `CS2-Predictor`.
3. Dacă te întreabă, acceptă `create a repository`. Lasă numele `CS2-Predictor`,
   bifează **Private** dacă vrei să fie ascuns.
4. Scrie un mesaj scurt (ex. „prima versiune") și apasă **Commit to main**.
5. Apasă **Publish repository**. Gata — codul e pe GitHub.

`.env`-ul cu cheile tale **nu** se urcă (e în `.gitignore`), deci ești în siguranță.

### Varianta B — linie de comandă

```bash
cd "calea/către/CS2-Predictor"
git init
git add .
git commit -m "prima versiune"
git branch -M main
git remote add origin https://github.com/<utilizatorul-tău>/CS2-Predictor.git
git push -u origin main
```

(Creezi întâi un repository gol pe github.com, fără README, și pui linkul lui la `origin`.)

---

## Pasul 2 — Deploy pe Streamlit Cloud

1. Intră pe **share.streamlit.io** și conectează-te cu același cont GitHub.
2. Apasă **Create app → Deploy a public app from GitHub**.
3. Completează:
   - **Repository**: `<utilizatorul-tău>/CS2-Predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Apasă **Deploy**. În 1–2 minute primești un link public (`...streamlit.app`).

---

## Pasul 3 — Adaugă cheile (secrete)

Ca să meargă stratul Claude și cotele, în pagina aplicației de pe Streamlit Cloud:
**Manage app → Settings → Secrets**, și lipește (format TOML):

```toml
ANTHROPIC_API_KEY = "cheia-ta-anthropic"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ODDS_API_KEY = "cheia-ta-odds-api"
```

Salvează. Aplicația repornește singură și citește cheile. Fără ele, aplicația tot
merge — doar fără briefing-ul scris de Claude.

---

## La prima deschidere

Aplicația vine cu un instantaneu de date (slate-ul din 28 mai + rezultatele), deci ai
ce vedea imediat. Ca să aduci meciuri noi, apasă pe pagina **Home** butonul
**Reîmprospătează + prezice** (durează câteva minute, fiindcă reia istoricul de pe
bo3.gg). Pe planul gratuit, dacă reîmprospătarea e prea grea, rulează întâi local și
împinge `data/clean` actualizat pe GitHub.

---

## Ce nu pot face eu

Push-ul pe GitHub și deploy-ul pe Streamlit Cloud cer autentificarea ta — nu mă pot
loga în conturile tale. Tot restul (codul, config-ul, secretele excluse, instantaneul
de date) e gata pregătit. Pot să te ghidez pas cu pas în browser dacă vrei.
