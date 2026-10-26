from __future__ import annotations
import os, re
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TimesFM AI Studio", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#090d13;--panel:#111720;--line:#26313f;--text:#edf2fa;--muted:#9aa9bb;--mint:#7de6ca;--lav:#b5a3ff}
html,body,[class*="css"]{font-family:Manrope,sans-serif}
.stApp{background:radial-gradient(ellipse at 50% -10%,#182738 0%,#0b1017 48%,#090d13 100%);color:var(--text)}
.block-container{padding-top:1rem;padding-bottom:1rem;max-width:1500px}
section[data-testid="stSidebar"]{background:#0b1119;border-right:1px solid #202b39}
.stButton>button,.stDownloadButton>button{border:1px solid #2b394a;border-radius:12px;background:#141e2a;color:#edf2fa;font-weight:650;min-height:39px}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--mint);color:var(--mint)}
.stTextInput input,.stTextArea textarea{background:#101722!important;border:1px solid #2b394a!important;border-radius:12px!important;color:#edf2fa!important}
div[data-testid="stChatMessage"]{background:transparent;border:0;padding:1rem .6rem}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){background:#111a25;border:1px solid #202e3e;border-radius:17px}
div[data-testid="stMetric"]{background:#111a25;border:1px solid #263343;border-radius:14px;padding:12px}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid #273343}
.stTabs [data-baseweb="tab"]{background:#111822;border-radius:10px 10px 0 0;padding:10px 16px}
.chat-header{display:flex;align-items:center;gap:12px;padding:12px 16px;border:1px solid #263343;border-radius:17px;background:#101720;margin-bottom:15px}
.avatar{width:42px;height:42px;background:linear-gradient(140deg,#1d4948,#252c4b);display:flex;align-items:center;justify-content:center;border-radius:14px;color:#8af0d7;font-size:18px;font-weight:800}
.chat-name{font-weight:800;font-size:16px}.muted{font-size:12px;color:#96a7ba}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#7de6ca;margin-right:6px}
.pill{font:11px 'DM Mono',monospace;color:#a5b8cc;border:1px solid #2a3747;border-radius:999px;padding:5px 9px}
.tip{border:1px solid #293747;background:#111923;border-radius:13px;padding:10px 12px;margin:6px 0;color:#dce7f3}
hr{border-color:#263343}
</style>
, unsafe_allow_html=True)

def init_state():
    defaults = {"messages":[{"role":"assistant","content":"Bienvenue dans TimesFM AI Studio. Charge un CSV ou utilise la démo, puis demande une analyse ou une prévision."}],
        "series":None,"dates":None,"series_name":"Série","forecast":None,"engine":"En attente","tasks":[]}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
init_state()

def demo():
    n=180; x=np.arange(n); rng=np.random.default_rng(42)
    y=35+0.05*x+5*np.sin(x*2*np.pi/7)+7*np.sin(x*2*np.pi/40)+rng.normal(0,1.8,n)
    return pd.Series(y,name="Valeur"),pd.date_range(end=pd.Timestamp.today().normalize(),periods=n,freq="D"),"Démo synthétique"

@st.cache_resource(show_spinner=False)
def load_model():
    import torch, timesfm
    torch.set_float32_matmul_precision("high")
    model=timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=1024,max_horizon=512,normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=False,fix_quantile_crossing=True))
    return model

def forecast(values,horizon,allow_fallback=True):
    from core import forecast_series
    # Use the cached model loader in the UI; the core handles validation and fallback.
    if allow_fallback:
        return forecast_series(values,horizon,model_loader=load_model)
    return forecast_series(values,horizon,model_loader=load_model)

def do_forecast(horizon):
    series=st.session_state.series
    if series is None: return "Aucune série chargée. Importe un CSV ou demande « charge la démo »."
    pred,bounds,engine=forecast(series.to_numpy(),horizon)
    dates=st.session_state.dates
    if dates is not None and len(dates)>1:
        delta=dates.to_series().diff().dropna().median()
        if pd.isna(delta) or delta<=pd.Timedelta(0): delta=pd.Timedelta(days=1)
        future=pd.DatetimeIndex([dates[-1]+delta*(i+1) for i in range(horizon)])
    else: future=np.arange(len(series),len(series)+horizon)
    st.session_state.forecast={"values":pred,"bounds":bounds,"dates":future,"horizon":horizon}
    st.session_state.engine=engine
    st.session_state.tasks.insert(0,{"action":f"Prévision {horizon} périodes","heure":datetime.now().strftime("%H:%M:%S"),"moteur":engine})
    st.session_state.tasks=st.session_state.tasks[:100]
    pct=(pred[-1]/series.iloc[-1]-1)*100 if series.iloc[-1] else 0
    return f"Prévision terminée sur {horizon} périodes. Dernière estimation : {pred[-1]:.3f} ({pct:+.1f}% par rapport à la dernière observation). Moteur : {engine}. Une prévision n'est pas une garantie."

def answer(prompt):
    text=prompt.lower()
    if any(w in text for w in ("aide","help","commande","capacités")):
        return "Actions disponibles : charge la démo, analyse les données, prévois 7/30/90 jours, ou importe un CSV. Le chat intégré exécute des commandes locales ; il ne s'agit pas encore d'un LLM généraliste."
    if any(w in text for w in ("démo","demo","exemple")):
        s,d,name=demo(); st.session_state.series=s; st.session_state.dates=d; st.session_state.series_name=name
        st.session_state.forecast=None
        return "Démo synthétique chargée : 180 observations journalières. Tu peux maintenant demander une analyse ou une prévision."
    if st.session_state.series is None:
        return "Aucune série chargée. Utilise l'import CSV dans la barre latérale ou demande « charge la démo »."
    if any(w in text for w in ("analyse","résume","resume","statistique","tendance")):
        y=st.session_state.series.to_numpy()
        delta=(y[-1]/y[0]-1)*100 if y[0] else 0
        return f"Analyse de « {st.session_state.series_name} » : {len(y)} points, moyenne {np.mean(y):.3f}, médiane {np.median(y):.3f}, minimum {np.min(y):.3f}, maximum {np.max(y):.3f}, variation début-fin {delta:+.1f}%. Moyenne des 14 dernières valeurs : {np.mean(y[-min(14,len(y)):]):.3f}."
    match=re.search(r"(\d{1,3})\s*(?:jours?|périodes?|periodes?|points?|valeurs?)",text)
    horizon=int(match.group(1)) if match else 30
    if any(w in text for w in ("prévoi","prevoi","prédit","predit","forecast","projection","projette","futur","calcul","lance")):
        return do_forecast(horizon)
    return "Essaie « analyse les données », « prévois 30 jours » ou « aide »."

with st.sidebar:
    st.markdown("## ◈ TimesFM")
    st.caption("AI Studio · Assistant de prévision")
    st.divider()
    if st.button("＋ Nouvelle conversation",use_container_width=True):
        st.session_state.messages=[{"role":"assistant","content":"Nouvelle conversation ouverte. Charge une série ou demande la démo pour commencer."}]
        st.rerun()
    st.markdown("#### Données")
    uploaded=st.file_uploader("Importer un CSV",type=["csv"],help="Une colonne numérique est nécessaire ; la date est facultative.")
    if uploaded is not None:
        try:
            df=pd.read_csv(uploaded)
            numeric=[c for c in df.columns if pd.to_numeric(df[c],errors="coerce").notna().sum()>=10]
            if not numeric: st.warning("Aucune colonne avec au moins 10 valeurs numériques.")
            else:
                value_col=st.selectbox("Colonne de valeurs",numeric)
                date_col=st.selectbox("Colonne date",["(index)"]+list(df.columns))
                if st.button("Utiliser ce fichier",use_container_width=True):
                    vals=pd.to_numeric(df[value_col],errors="coerce")
                    if date_col!="(index)":
                        parsed=pd.to_datetime(df[date_col],errors="coerce")
                        clean=pd.DataFrame({"value":vals,"date":parsed}).dropna().sort_values("date")
                        values=clean["value"].astype(float).reset_index(drop=True)
                        parsed_dates=pd.DatetimeIndex(clean["date"])
                    else:
                        values=vals.dropna().astype(float).reset_index(drop=True); parsed_dates=None
                    if len(values)<10: st.error("Il faut au moins 10 valeurs valides.")
                    elif not np.isfinite(values.to_numpy()).all(): st.error("Valeurs non finies détectées.")
                    else:
                        st.session_state.series=values; st.session_state.dates=parsed_dates; st.session_state.series_name=value_col
                        st.session_state.forecast=None
                        st.session_state.messages.append({"role":"assistant","content":f"Fichier chargé : **{value_col}**, {len(values)} observations valides. Tu peux demander une analyse ou une prévision."})
                        st.success(f"{len(values)} observations prêtes.")
        except Exception as exc: st.error(f"Lecture CSV impossible : {exc}")
    if st.button("Charger une démo",use_container_width=True):
        s,d,name=demo(); st.session_state.series=s; st.session_state.dates=d; st.session_state.series_name=name; st.session_state.forecast=None
        st.session_state.messages.append({"role":"assistant","content":"Je viens de charger une série de démonstration synthétique de 180 observations quotidiennes. Essaie « analyse cette série » ou « prévois 30 jours »."})
        st.rerun()
    st.divider()
    st.markdown("#### Raccourcis")
    for label,cmd in [("Analyser la série","Analyse les données"),("Prévoir 7 jours","Prévois 7 jours"),("Prévoir 30 jours","Prévois 30 jours"),("Prévoir 90 jours","Prévois 90 jours"),("Comment utiliser ?","Aide")]:
        if st.button(label,key="quick_"+label,use_container_width=True):
            st.session_state.messages.append({"role":"user","content":cmd})
            with st.spinner("L’assistant traite la demande…"): response=answer(cmd)
            st.session_state.messages.append({"role":"assistant","content":response}); st.rerun()
    st.divider()
    st.markdown("#### État du moteur")
    st.markdown(f'<div class="tip"><span class="status-dot"></span><b>TimesFM 2.5</b><div class="muted" style="margin-top:5px">{st.session_state.engine}</div></div>',unsafe_allow_html=True)
    st.caption("Le chat exécute des commandes locales ; il n’est pas encore relié à un LLM généraliste.")

st.markdown('<div class="chat-header"><div class="avatar">◈</div><div style="flex:1"><div class="chat-name">TimesFM Assistant</div><div class="muted"><span class="status-dot"></span>Prêt à analyser tes séries temporelles</div></div><div class="pill">LOCAL WORKSPACE</div></div>',unsafe_allow_html=True)
chat_tab,data_tab,history_tab=st.tabs(["✦  Conversation","⌁  Données & prévisions","▤  Activité"])
with chat_tab:
    if st.session_state.series is not None:
        c1,c2,c3=st.columns(3)
        c1.metric("Série active",st.session_state.series_name)
        c2.metric("Observations",len(st.session_state.series))
        c3.metric("Prévision",f"{st.session_state.forecast['horizon']} points" if st.session_state.forecast else "Pas encore lancée")
    else:
        st.markdown("### Que souhaites-tu faire ?")
        st.caption("Commence par une commande ou charge tes données dans le panneau à gauche.")
        col1,col2,col3=st.columns(3)
        with col1:
            if st.button("📈  Charger la démo",use_container_width=True):
                s,d,name=demo(); st.session_state.series=s; st.session_state.dates=d; st.session_state.series_name=name; st.rerun()
        with col2:
            if st.button("🔎  Analyse statistique",use_container_width=True):
                cmd="Analyse les données"; st.session_state.messages.append({"role":"user","content":cmd}); st.session_state.messages.append({"role":"assistant","content":answer(cmd)}); st.rerun()
        with col3:
            if st.button("🗓️  Prévision 30 jours",use_container_width=True):
                cmd="Prévois 30 jours"; st.session_state.messages.append({"role":"user","content":cmd}); st.session_state.messages.append({"role":"assistant","content":answer(cmd)}); st.rerun()
    st.markdown("---")
    for message in st.session_state.messages:
        with st.chat_message(message["role"],avatar="◈" if message["role"]=="assistant" else "●"):
            st.markdown(message["content"])
    prompt=st.chat_input("Écris ton message… (ex. : analyse les données, prévois 30 jours)")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.spinner("L’assistant réfléchit et exécute la commande…"):
            try: response=answer(prompt)
            except Exception as exc: response=f"Je n’ai pas pu terminer cette action : {type(exc).__name__}: {exc}"
        st.session_state.messages.append({"role":"assistant","content":response})
        st.rerun()
with data_tab:
    st.markdown("### Données et résultats")
    if st.session_state.series is None:
        st.info("Aucune série chargée. Utilise le bouton « Charger une démo » ou importe un CSV à gauche.")
    else:
        series=st.session_state.series
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Moyenne",f"{series.mean():.3f}"); m2.metric("Minimum",f"{series.min():.3f}")
        m3.metric("Maximum",f"{series.max():.3f}"); m4.metric("Dernière valeur",f"{series.iloc[-1]:.3f}")
        fig=go.Figure()
        history_x=st.session_state.dates if st.session_state.dates is not None else np.arange(len(series))
        fig.add_trace(go.Scatter(x=history_x,y=series,name="Historique",line=dict(color="#b5a3ff",width=2)))
        f=st.session_state.forecast
        if f:
            fig.add_trace(go.Scatter(x=f["dates"],y=f["values"],name="Prévision",line=dict(color="#7de6ca",width=2.5)))
            if f["bounds"] is not None:
                fig.add_trace(go.Scatter(x=f["dates"],y=f["bounds"][:,1],line=dict(width=0),showlegend=False,hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=f["dates"],y=f["bounds"][:,0],line=dict(width=0),fill="tonexty",fillcolor="rgba(125,230,202,.12)",name="Bande indicative"))
        fig.update_layout(template="plotly_dark",height=430,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=15,b=5),legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)
        if f:
            out=pd.DataFrame({"date":np.asarray(f["dates"]).astype(str),"forecast":f["values"]})
            if f["bounds"] is not None: out["lower"]=f["bounds"][:,0]; out["upper"]=f["bounds"][:,1]
            st.dataframe(out,use_container_width=True,hide_index=True)
            st.download_button("Télécharger le CSV de prévision",out.to_csv(index=False).encode("utf-8"),file_name="timesfm_forecast.csv",mime="text/csv")
with history_tab:
    st.markdown("### Activité de cette session")
    if st.session_state.tasks: st.dataframe(pd.DataFrame(st.session_state.tasks),use_container_width=True,hide_index=True)
    else: st.info("Les analyses et prévisions exécutées apparaîtront ici.")
st.markdown('<div class="muted" style="text-align:center;padding:16px 0">TimesFM AI Studio · Chat séries temporelles · Le mode de secours est indiqué dans le statut du moteur</div>',unsafe_allow_html=True)


st.markdown('<div class="mono" style="margin-top:22px">TIMESFM AI STUDIO · Interface OpenClaw-inspired · Chat local à commandes (pas de LLM généraliste configuré)</div>',unsafe_allow_html=True)
