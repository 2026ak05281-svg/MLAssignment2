"""
Telco Customer Churn -- classification model evaluation app.

Loads five pre-trained classifiers and scores them on an uploaded test CSV.
Models are never retrained; they are loaded from model/*.joblib.

Run locally:  streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'model')
DEFAULT_TEST = os.path.join(BASE, 'test_data.csv')

TARGET = 'Churn'
METRIC_COLS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'k_nearest_neighbors.joblib',
    'Naive Bayes': 'naive_bayes_gaussian.joblib',
    'Support Vector Machine': 'support_vector_machine.joblib',
}

st.set_page_config(page_title='Telco Churn — Model Evaluation',
                   page_icon=':bar_chart:', layout='wide')


@st.cache_resource
def load_models():
    """Load the five pre-trained pipelines. Never retrains."""
    models, failures = {}, {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            failures[name] = f'file not found: {fname}'
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as exc:
            # Usually a scikit-learn version mismatch against the pickled model.
            failures[name] = f'{type(exc).__name__}: {exc}'
    return models, failures


@st.cache_data
def load_training_metadata():
    path = os.path.join(MODEL_DIR, 'metrics.json')
    return json.load(open(path)) if os.path.exists(path) else None


def prepare(df):
    """Validate the uploaded frame and split it into X and a 0/1 y."""
    if TARGET not in df.columns:
        return None, None, f"Uploaded CSV must contain the target column '{TARGET}'."

    y_raw = df[TARGET]
    if y_raw.dtype == object or str(y_raw.dtype).startswith('str'):
        mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0,
                   'churn': 1, 'churned': 1, 'retained': 0}
        y = y_raw.astype(str).str.strip().str.lower().map(mapping)
        if y.isna().any():
            bad = sorted(y_raw[y.isna()].astype(str).unique())[:5]
            return None, None, f'Could not interpret target values: {bad}'
    else:
        y = pd.to_numeric(y_raw, errors='coerce')
        if y.isna().any():
            return None, None, 'Target column contains non-numeric values.'

    X = df.drop(columns=[TARGET]).copy()

    # TotalCharges arrives as text with blanks for tenure-0 customers.
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0.0)

    return X, y.astype(int), None


def score_model(pipe, X, y):
    """Compute all six required metrics for one fitted pipeline."""
    y_pred = pipe.predict(X)
    try:
        y_proba = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
    except Exception:
        y_proba, auc = None, np.nan
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'AUC': auc,
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y, y_pred),
    }, y_pred, y_proba


MODELS, LOAD_FAILURES = load_models()
META = load_training_metadata()

# ----------------------------------------------------------------- sidebar
st.sidebar.title('Telco Churn Evaluation')

if not MODELS:
    st.error(f'No models could be loaded from {MODEL_DIR}.')
    for name, reason in LOAD_FAILURES.items():
        st.text(f'  {name}: {reason}')
    st.info('Run `python train_models.py` to regenerate the model files, and '
            'confirm the scikit-learn version matches requirements.txt.')
    st.stop()

st.sidebar.success(f'{len(MODELS)} pre-trained models loaded')

if LOAD_FAILURES:
    st.sidebar.warning(f'{len(LOAD_FAILURES)} model(s) failed to load')
    with st.sidebar.expander('Details'):
        for name, reason in LOAD_FAILURES.items():
            st.text(f'{name}: {reason}')

st.sidebar.markdown('### 1. Test data')
uploaded = st.sidebar.file_uploader(
    'Upload test CSV', type=['csv'],
    help=f"Must contain all input features and the '{TARGET}' target column.")

use_bundled = st.sidebar.checkbox(
    'Use bundled test_data.csv', value=True,
    help='Uncheck to rely only on your own upload.')

st.sidebar.markdown('### 2. Model')
model_choice = st.sidebar.selectbox(
    'Select model for detailed view', list(MODELS.keys()))

st.sidebar.markdown('---')
st.sidebar.caption('Models are loaded from disk and never retrained.')

# ------------------------------------------------------------- load frame
source_label = None
df = None

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_label = f'uploaded file — {uploaded.name}'
elif use_bundled and os.path.exists(DEFAULT_TEST):
    df = pd.read_csv(DEFAULT_TEST)
    source_label = 'bundled test_data.csv'

st.title('Telco Customer Churn — Model Evaluation')

if df is None:
    st.info('Upload a test CSV from the sidebar, or tick "Use bundled test_data.csv".')
    st.stop()

X, y, err = prepare(df)
if err:
    st.error(err)
    st.write('Columns found in your file:', list(df.columns))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Rows', f'{len(X):,}')
c2.metric('Features', X.shape[1])
c3.metric('Churn rate', f'{y.mean():.1%}')
c4.metric('Models', len(MODELS))
st.caption(f'Data source: {source_label}')

with st.expander('Preview test data'):
    st.dataframe(df.head(20), use_container_width=True)

# -------------------------------------------------- score every model live
rows, preds, probas, errors = {}, {}, {}, {}
for name, pipe in MODELS.items():
    try:
        m, y_pred, y_proba = score_model(pipe, X, y)
        rows[name] = m
        preds[name] = y_pred
        probas[name] = y_proba
    except Exception as exc:
        errors[name] = str(exc)

if errors:
    st.warning('Some models could not score this file:')
    for n, e in errors.items():
        st.text(f'  {n}: {e}')

if not rows:
    st.error('No model could score the uploaded data. Check that the feature '
             'columns match the training schema.')
    st.stop()

results = pd.DataFrame(rows).T[METRIC_COLS]

tab1, tab2, tab3, tab4 = st.tabs(
    ['Evaluation Metrics', 'Confusion Matrix & Report', 'ROC Curves', 'Model Details'])

# ------------------------------------------------------ tab 1: metrics
with tab1:
    st.subheader('Evaluation metrics — all models on this test data')
    st.dataframe(
        results.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True)

    best = results['MCC'].idxmax()
    baseline = max(y.mean(), 1 - y.mean())
    c1, c2, c3 = st.columns(3)
    c1.metric('Best model (MCC)', best, f"{results.loc[best, 'MCC']:.4f}")
    c2.metric('Best accuracy', results['Accuracy'].idxmax(),
              f"{results['Accuracy'].max():.4f}")
    c3.metric('Majority-class baseline', f'{baseline:.4f}',
              help='Accuracy from always predicting the majority class.')

    st.subheader('Compare a single metric')
    metric = st.selectbox('Metric', METRIC_COLS)
    ranked = results[metric].sort_values()
    fig = px.bar(x=ranked.values, y=ranked.index, orientation='h',
                 labels={'x': metric, 'y': ''}, text=ranked.values,
                 color=ranked.values, color_continuous_scale='Viridis')
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(showlegend=False, xaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('All metrics side by side')
    long = results.reset_index().melt(id_vars='index', var_name='Metric',
                                      value_name='Score')
    long.columns = ['Model', 'Metric', 'Score']
    st.plotly_chart(px.bar(long, x='Metric', y='Score', color='Model',
                           barmode='group', range_y=[0, 1]),
                    use_container_width=True)

    st.download_button('Download metrics as CSV',
                       results.round(4).to_csv().encode(),
                       'evaluation_metrics.csv', 'text/csv')

# --------------------------------------- tab 2: confusion + report
with tab2:
    st.subheader(f'Confusion matrix — {model_choice}')
    if model_choice not in preds:
        st.error(f'{model_choice} could not score this data.')
    else:
        cm = confusion_matrix(y, preds[model_choice])
        tn, fp, fn, tp = cm.ravel()

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                            labels={'x': 'Predicted', 'y': 'Actual'},
                            x=['No churn', 'Churn'], y=['No churn', 'Churn'])
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric('True negatives', int(tn))
            st.metric('False positives', int(fp), help='Loyal customers wrongly flagged')
            st.metric('False negatives', int(fn), help='Churners the model missed')
            st.metric('True positives', int(tp))
            st.metric('Churners caught', f'{tp} of {tp + fn}')

        st.subheader(f'Classification report — {model_choice}')
        rep = classification_report(y, preds[model_choice],
                                    target_names=['No churn', 'Churn'],
                                    digits=4, zero_division=0)
        st.code(rep, language='text')

        st.subheader('Metrics for this model')
        st.dataframe(results.loc[[model_choice]].style.format('{:.4f}'),
                     use_container_width=True)

    st.subheader('Confusion matrices — all models')
    cols = st.columns(len(preds))
    for col, (name, yp) in zip(cols, preds.items()):
        with col:
            cm = confusion_matrix(y, yp)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                            x=['No', 'Yes'], y=['No', 'Yes'])
            fig.update_layout(title={'text': name, 'font': {'size': 12}},
                              height=250, margin={'l': 5, 'r': 5, 't': 40, 'b': 5},
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------ tab 3: ROC
with tab3:
    st.subheader('ROC curves on this test data')
    fig = go.Figure()
    for name, proba in probas.items():
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y, proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"{name} (AUC={results.loc[name, 'AUC']:.4f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                             line={'dash': 'dash', 'color': 'grey'}))
    fig.update_layout(xaxis_title='False positive rate',
                      yaxis_title='True positive rate', height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        'AUC measures how well a model **ranks** churners above non-churners and '
        'is independent of the 0.5 decision threshold. A model can therefore show '
        'a strong AUC alongside weak accuracy.')

# -------------------------------------------------- tab 4: model details
with tab4:
    st.subheader('Model configuration')
    info = []
    for name, pipe in MODELS.items():
        clf = pipe.named_steps.get('classifier') or pipe.steps[-1][1]
        info.append({'Model': name, 'Estimator': clf.__class__.__name__,
                     'Parameters': str(clf)[:120]})
    st.dataframe(pd.DataFrame(info), hide_index=True, use_container_width=True)

    if META:
        st.subheader('Training-time reference metrics')
        st.caption('Scores recorded when the models were trained, for comparison '
                   'with the live scores above.')
        ref = pd.DataFrame(META['results']).T[
            ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']]
        ref.columns = METRIC_COLS
        st.dataframe(ref.style.format('{:.4f}'), use_container_width=True)

        d = META['dataset']
        st.markdown(f"""
**Training setup**

- Source rows: {d['rows']:,} · features: {d['features']}
- Stratified split: {d['train_rows']:,} train / {d['test_rows']:,} test, `random_state=42`
- `StandardScaler` on {len(d['numeric_cols'])} numeric columns,
  `OneHotEncoder(drop='first')` on {len(d['categorical_cols'])} categorical columns
- Preprocessing is bundled inside each `Pipeline`, fitted on training data only
""")

    st.subheader('Expected input schema')
    if META:
        schema = META['schema']
        st.dataframe(pd.DataFrame([
            {'Column': c, 'Type': m['type'],
             'Allowed / range': (', '.join(map(str, m['options']))[:70]
                                 if m['type'] == 'categorical'
                                 else f"{m['min']} to {m['max']}")}
            for c, m in schema.items()
        ]), hide_index=True, use_container_width=True)
    st.caption(f"Your uploaded CSV must also include the '{TARGET}' target column "
               "(Yes/No or 1/0).")
