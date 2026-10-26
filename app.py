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
:root{--bg:#080d14;--panel:#101722;--line:#273447;--text:#eaf0f8;--muted:#93a3b7;--mint:#79e6cb;--violet:#b6a0ff}
html,body,[class*="css"]{font-family:Manrope,sans-serif}
.stApp{background:radial-gradient(ellipse at 60% -15%,#172d43 0%,#090f18 48%,#080d14 100%);color:var(--text)}
.block-container{padding-top:1.2rem;max-width:1600px}
section[data-testid="stSidebar"]{background:#0b111a;border-right:1px solid #202d3d}
div[data-testid="stMetric"]{background:#111b28;border:1px solid #28374b;border-radius:14px;padding:12px}
.stButton>button,.stDownloadButton>button{border:1px solid #2b3b50;border-radius:11px;background:#131f2d;color:#eaf0f8;font-weight:650;min-height:40px}
.stButton>button:hover{border-color:var(--mint);color:var(--mint)}
.stTextInput input,.stTextArea textarea{background:#0d1520!important;border:1px solid #2b3a4d!important;border-radius:12px!important;color:#eaf0f8!important}
div[data-testid="stChatMessage"]{background:#111a27;border:1px solid #263447;border-radius:15px}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border:1px solid #253448;border-radius:15px;background:#0f1723;margin-bottom:16px}
.brand{font-weight:800;font-size:20px;letter-spacing:-.6px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#79e6cb;margin-right:8px}
.mono{font-family:'DM Mono',monospace;color:#8fa4bb;font-size:11px;letter-spacing:.6px}
.panel{background:#101824;border:1px solid #26364a;border-radius:16px;padding:16px}
.agent{display:flex;align-items:center;gap:10px;padding:10px;border-radius:12px;background:#121d2b;border:1px solid #26364a;margin:7px 0}
.agent-icon{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#203a45;color:#79e6cb;font-weight:800}
.small{font-size:12px;color:#94a5b9}
</style>""", unsafe_allow_html=True)

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
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=1 or len(values)<10 or not np.isfinite(values).all():
        raise ValueError("La série doit contenir au moins 10 valeurs numériques finies.")
    horizon=max(1,min(512,int(horizon)))
    try:
        model=load_model()
        preds,quantiles=model.forecast(horizon=horizon,inputs=[values])
        pred=np.asarray(preds[0],dtype=float)[:horizon]
        if len(pred)!=horizon or not np.isfinite(pred).all(): raise ValueError("Le modèle a renvoyé des valeurs invalides.")
        bounds=None
        if quantiles is not None:
            q=np.asarray(quantiles)
            if q.ndim==3 and q.shape[1]>=horizon and q.shape[2]>=2:
                q=q[0,:horizon,:]
                bounds=np.column_stack((q[:,1],q[:,-1])) if q.shape[1]>=9 else None
        return pred,bounds,"TimesFM 2.5 — modèle officiel"
    except Exception as exc:
        if not allow_fallback: raise
        window=min(len(values),max(10,min(60,len(values)//2)))
        y=values[-window:]; x=np.arange(window)
        slope,intercept=np.polyfit(x,y,1)
        pred=slope*np.arange(window,window+horizon)+intercept
        resid=y-(slope*x+intercept); spread=max(float(np.std(resid)),float(np.mean(np.abs(y)))*.03,1e-6)
        radius=1.28*spread*np.sqrt(np.arange(1,horizon+1))
        return pred,np.column_stack((pred-radius,pred+radius)),f"Mode statistique de secours — {type(exc).__name__}: {str(exc)[:120]}"

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
    st.markdown('<div class="mono">WORKSPACE / LOCAL AGENT</div>',unsafe_allow_html=True)
    st.markdown("## ◈ TimesFM AI Studio")
    st.caption("Assistant de séries temporelles")
    st.divider()
    uploaded=st.file_uploader("Importer un CSV",type=["csv"],help="CSV avec une colonne numérique et éventuellement une colonne date.")
    if uploaded is not None:
        try:
            df=pd.read_csv(uploaded)
            numeric=[c for c in df.columns if pd.to_numeric(df[c],errors="coerce").notna().sum()>=10]
            if not numeric: st.warning("Aucune colonne avec au moins 10 valeurs numériques.")
            else:
                value_col=st.selectbox("Colonne de valeurs",numeric)
                date_choices=["(index)"]+list(df.columns)
                date_col=st.selectbox("Colonne date (facultatif)",date_choices)
                if st.button("Charger le CSV",use_container_width=True):
                    vals=pd.to_numeric(df[value_col],errors="coerce")
                    if date_col!="(index)":
                        parsed=pd.to_datetime(df[date_col],errors="coerce")
                        clean=pd.DataFrame({"value":vals,"date":parsed}).dropna()
                        clean=clean.sort_values("date")
                        values=clean["value"].astype(float).reset_index(drop=True)
                        parsed_dates=pd.DatetimeIndex(clean["date"])
                    else:
                        values=vals.dropna().astype(float).reset_index(drop=True); parsed_dates=None
                    if len(values)<10: st.error("Il faut au moins 10 valeurs valides après nettoyage.")
                    elif not np.isfinite(values.to_numpy()).all(): st.error("Le fichier contient des valeurs non finies.")
                    else:
                        st.session_state.series=values; st.session_state.dates=parsed_dates; st.session_state.series_name=value_col
                        st.session_state.forecast=None
                        st.success(f"{len(values)} observations chargées.")
        except Exception as exc: st.error(f"Lecture CSV impossible : {exc}")
    if st.button("Charger la démo",use_container_width=True):
        s,d,name=demo(); st.session_state.series=s; st.session_state.dates=d; st.session_state.series_name=name; st.session_state.forecast=None; st.rerun()
    st.divider()
    st.markdown("**Moteur**")
    st.markdown('<div class="agent"><div class="agent-icon">TF</div><div><b>TimesFM 2.5</b><div class="small">Forecast engine</div></div></div>',unsafe_allow_html=True)
    st.caption(st.session_state.engine)
    for n in (7,30,90):
        if st.button(f"Prévoir {n} jours",key=f"side_{n}",use_container_width=True):
            prompt=f"Prévois {n} jours"
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.spinner("Calcul de la prévision…"): response=do_forecast(n)
            st.session_state.messages.append({"role":"assistant","content":response}); st.rerun()
    if st.button("Nouvelle conversation",use_container_width=True):
        st.session_state.messages=[{"role":"assistant","content":"Nouvelle session. Charge un CSV ou demande la démo."}]; st.rerun()

st.markdown('<div class="topbar"><div><span class="dot"></span><span class="brand">TimesFM AI Studio</span><div class="mono">PERSONAL FORECASTING WORKSPACE</div></div><div class="mono">LOCAL MODE · CHAT + TOOLS</div></div>',unsafe_allow_html=True)
a,b,c,d=st.columns(4)
a.metric("Série active",st.session_state.series_name if st.session_state.series is not None else "Aucune")
b.metric("Observations",len(st.session_state.series) if st.session_state.series is not None else "—")
c.metric("État", "Données prêtes" if st.session_state.series is not None else "En attente")
d.metric("Tâches",len(st.session_state.tasks))
chat,board,history=st.tabs(["◈ Chat agent","⌁ Dashboard","▤ Historique"])
with chat:
    left,right=st.columns([1.8,1],gap="large")
    with left:
        st.markdown("### Conversation")
        for message in st.session_state.messages:
            with st.chat_message(message["role"],avatar="◈" if message["role"]=="assistant" else "●"):
                st.markdown(message["content"])
        prompt=st.chat_input("Demande une analyse ou une prévision…")
        if prompt:
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.spinner("Exécution de la demande…"): response=answer(prompt)
            st.session_state.messages.append({"role":"assistant","content":response}); st.rerun()
    with right:
        st.markdown("### Agent runtime")
        st.markdown('<div class="panel"><div class="mono">ACTIVE AGENT</div><h3 style="margin:8px 0">TimesFM Analyst</h3><p class="small">Analyse de séries · Prévision · Export</p><span class="mono">STATUS: READY</span></div>',unsafe_allow_html=True)
        st.markdown("#### Outils disponibles")
        for title,desc in [("CSV Loader","Import et nettoyage"),("Forecast Engine","TimesFM et fallback indiqué"),("Statistics","Résumé de la série"),("Chart Builder","Graphique interactif"),("CSV Export","Téléchargement des prévisions")]:
            st.markdown(f'<div class="agent"><div class="agent-icon">↗</div><div><b>{title}</b><div class="small">{desc}</div></div></div>',unsafe_allow_html=True)
        for label,command in [("Analyser les données","Analyse les données"),("Prévoir 30 jours","Prévois 30 jours"),("Aide","Aide")]:
            if st.button(label,key="suggest_"+label,use_container_width=True):
                st.session_state.messages.append({"role":"user","content":command})
                with st.spinner("Exécution…"): response=answer(command)
                st.session_state.messages.append({"role":"assistant","content":response}); st.rerun()
with board:
    st.markdown("### Dashboard de prévision")
    if st.session_state.series is None: st.info("Charge un CSV ou les données de démonstration pour commencer.")
    else:
        series=st.session_state.series
        m1,m2,m3,m4=st.columns(4); m1.metric("Moyenne",f"{series.mean():.3f}"); m2.metric("Minimum",f"{series.min():.3f}"); m3.metric("Maximum",f"{series.max():.3f}"); m4.metric("Dernière valeur",f"{series.iloc[-1]:.3f}")
        fig=go.Figure()
        history_x=st.session_state.dates if st.session_state.dates is not None else np.arange(len(series))
        fig.add_trace(go.Scatter(x=history_x,y=series,name="Historique",line=dict(color="#9aafff",width=2)))
        f=st.session_state.forecast
        if f:
            fig.add_trace(go.Scatter(x=f["dates"],y=f["values"],name="Prévision",line=dict(color="#79e6cb",width=2.5)))
            if f["bounds"] is not None:
                fig.add_trace(go.Scatter(x=f["dates"],y=f["bounds"][:,1],line=dict(width=0),showlegend=False,hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=f["dates"],y=f["bounds"][:,0],line=dict(width=0),fill="tonexty",fillcolor="rgba(121,230,203,.13)",name="Bande indicative"))
        fig.update_layout(template="plotly_dark",height=440,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=8,r=8,t=20,b=8),legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)
        if f:
            out=pd.DataFrame({"date":np.asarray(f["dates"]).astype(str),"forecast":f["values"]})
            if f["bounds"] is not None: out["lower"]=f["bounds"][:,0]; out["upper"]=f["bounds"][:,1]
            st.dataframe(out,use_container_width=True,hide_index=True)
            st.download_button("Exporter la prévision CSV",out.to_csv(index=False).encode("utf-8"),file_name="timesfm_forecast.csv",mime="text/csv")
with history:
    st.markdown("### Historique des tâches")
    if st.session_state.tasks: st.dataframe(pd.DataFrame(st.session_state.tasks),use_container_width=True,hide_index=True)
    else: st.info("Aucune tâche exécutée dans cette session.")
st.markdown('<div class="mono" style="margin-top:22px">TIMESFM AI STUDIO · Interface OpenClaw-inspired · Chat local à commandes (pas de LLM généraliste configuré)</div>',unsafe_allow_html=True)
