"""
🎗️ Breast Cancer Prediction Dashboard
An interactive Streamlit dashboard for predicting breast cancer diagnosis
using a Decision Tree Classifier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎗️ Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #FF8E8E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .malignant {
        background: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
        color: white;
    }
    .benign {
        background: linear-gradient(135deg, #2ed573 0%, #7bed9f 100%);
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== PATHS ====================
PROJ_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJ_ROOT / "models"

# ==================== FEATURE DEFINITIONS ====================
FEATURE_GROUPS = {
    "Mean Values": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
    ],
    "Standard Error Values": [
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se"
    ],
    "Worst Values": [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    """Load the breast cancer dataset."""
    df = pd.read_csv(RAW_DATA_DIR / "breast-cancer-dataset.csv")
    df.drop("id", axis=1, inplace=True)
    return df

@st.cache_resource
def train_model(df):
    """Train a Decision Tree model for the dashboard."""
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with tuned parameters
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        min_samples_split=0.01,
        min_samples_leaf=0.01,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # Get predictions for metrics
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)
    
    return clf, scaler, X.columns.tolist(), (X_test_scaled, y_test, y_pred, y_pred_proba)

def get_feature_stats(df):
    """Get min, max, mean for each feature."""
    stats = {}
    for col in ALL_FEATURES:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    return stats

# ==================== SAMPLE DATA ====================
def get_sample_malignant(df):
    """Get a sample malignant case."""
    malignant = df[df["diagnosis"] == "M"].iloc[0]
    return {col: malignant[col] for col in ALL_FEATURES}

def get_sample_benign(df):
    """Get a sample benign case."""
    benign = df[df["diagnosis"] == "B"].iloc[0]
    return {col: benign[col] for col in ALL_FEATURES}

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎗️ Breast Cancer Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict breast cancer diagnosis using Decision Tree Classifier</p>', unsafe_allow_html=True)
    
    # Load data and model
    try:
        df = load_data()
        clf, scaler, feature_names, test_data = train_model(df)
        feature_stats = get_feature_stats(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the dataset exists at: data/raw/breast-cancer-dataset.csv")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Model Insights", "🔬 Data Explorer", "ℹ️ About"])
    
    # ==================== TAB 1: PREDICTION ====================
    with tab1:
        col_sidebar, col_main = st.columns([1, 2])
        
        with col_sidebar:
            st.markdown("### 📝 Input Features")
            
            # Quick Demo Buttons
            st.markdown("#### 💡 Quick Demo")
            demo_col1, demo_col2, demo_col3 = st.columns(3)
            
            with demo_col1:
                if st.button("🔴 Malignant", use_container_width=True):
                    sample = get_sample_malignant(df)
                    for f, val in sample.items():
                        st.session_state[f"slider_{f}"] = val
                    st.rerun()
            
            with demo_col2:
                if st.button("🟢 Benign", use_container_width=True):
                    sample = get_sample_benign(df)
                    for f, val in sample.items():
                        st.session_state[f"slider_{f}"] = val
                    st.rerun()

            with demo_col3:
                if st.button("🔄 Reset", use_container_width=True):
                    for f in ALL_FEATURES:
                        st.session_state[f"slider_{f}"] = feature_stats[f]["mean"]
                    st.rerun()
            
            st.divider()
            
            # Feature Input Sliders
            input_values = {}
            
            for group_name, features in FEATURE_GROUPS.items():
                with st.expander(f"📌 {group_name}", expanded=(group_name == "Mean Values")):
                    for feature in features:
                        stats = feature_stats[feature]
                        # Use a more user-friendly label
                        label = feature.replace("_", " ").title()
                        input_values[feature] = st.slider(
                            label,
                            min_value=stats["min"],
                            max_value=stats["max"],
                            value=stats["mean"],
                            key=f"slider_{feature}"
                        )
        
        with col_main:
            st.markdown("### 🎯 Prediction Result")
            
            # Prepare input for prediction
            # Ensure columns are in the exact same order as used during training
            input_df = pd.DataFrame([input_values])[feature_names]
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = clf.predict(input_scaled)[0]
            probability = clf.predict_proba(input_scaled)[0]
            
            # Display prediction
            if prediction == "M":
                prob_display = probability[1] * 100
                st.markdown(f"""
                <div class="prediction-box malignant">
                    <h2>🔴 MALIGNANT</h2>
                    <p style="font-size: 1.5rem;">Confidence: {prob_display:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.warning("⚠️ This prediction suggests the tumor may be malignant. Please consult a medical professional.")
            else:
                prob_display = probability[0] * 100
                st.markdown(f"""
                <div class="prediction-box benign">
                    <h2>🟢 BENIGN</h2>
                    <p style="font-size: 1.5rem;">Confidence: {prob_display:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ This prediction suggests the tumor is likely benign.")
            
            # Probability gauge
            st.markdown("#### Probability Distribution")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Malignancy Probability"},
                gauge={
                    'axis': {'range': [0, 100], 'ticksuffix': '%'},
                    'bar': {'color': "#ff4757"},
                    'steps': [
                        {'range': [0, 30], 'color': "#7bed9f"},
                        {'range': [30, 70], 'color': "#ffa502"},
                        {'range': [70, 100], 'color': "#ff6b81"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature importance for this prediction
            st.markdown("#### 🔍 Key Factors for This Prediction")
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig_imp = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='RdYlGn_r'
            )
            fig_imp.update_layout(
                height=350,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    
    # ==================== TAB 2: MODEL INSIGHTS ====================
    with tab2:
        st.markdown("### 📊 Model Performance Metrics")
        
        X_test, y_test, y_pred, y_pred_proba = test_data
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Precision", f"{precision:.2%}")
        with col3:
            st.metric("Recall", f"{recall:.2%}")
        with col4:
            st.metric("F1 Score", f"{f1:.2%}")
        
        st.divider()
        
        col_cm, col_roc = st.columns(2)
        
        with col_cm:
            st.markdown("#### Confusion Matrix")
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Benign', 'Malignant'],
                y=['Benign', 'Malignant'],
                color_continuous_scale='RdYlGn_r',
                text_auto=True
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col_roc:
            st.markdown("#### ROC Curve")
            # Get numeric labels
            y_test_numeric = (y_test == 'M').astype(int)
            fpr, tpr, _ = roc_curve(y_test_numeric, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                mode='lines',
                line=dict(color='#ff4757', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(color='gray', width=2, dash='dash')
            ))
            fig_roc.update_layout(
                height=400,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(x=0.5, y=0.1)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Feature Importance
        st.markdown("#### 🎯 Complete Feature Importance")
        all_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_all_imp = px.bar(
            all_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_all_imp.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig_all_imp, use_container_width=True)
    
    # ==================== TAB 3: DATA EXPLORER ====================
    with tab3:
        st.markdown("### 🔬 Dataset Explorer")
        
        # Dataset stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Malignant Cases", len(df[df["diagnosis"] == "M"]))
        with col3:
            st.metric("Benign Cases", len(df[df["diagnosis"] == "B"]))
        
        st.divider()
        
        # Diagnosis distribution
        col_dist, col_corr = st.columns(2)
        
        with col_dist:
            st.markdown("#### Diagnosis Distribution")
            fig_dist = px.pie(
                df,
                names='diagnosis',
                color='diagnosis',
                color_discrete_map={'M': '#ff4757', 'B': '#2ed573'},
                hole=0.4
            )
            fig_dist.update_traces(textinfo='percent+label+value')
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col_corr:
            st.markdown("#### Feature Correlation Heatmap")
            # Select top important features for correlation
            top_features = all_importance.tail(8)['Feature'].tolist()
            corr_matrix = df[top_features].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.divider()
        
        # Sample data table
        st.markdown("#### 📋 Sample Data")
        st.markdown("Click on a row to load those values into the prediction form!")
        
        # Show first 20 rows
        display_df = df.head(20).reset_index(drop=True)
        display_df.insert(0, 'Index', range(1, len(display_df) + 1))
        st.dataframe(display_df, use_container_width=True, height=400)
    
    # ==================== TAB 4: ABOUT ====================
    with tab4:
        st.markdown("""
        ### ℹ️ About This Project
        
        This dashboard demonstrates a **Decision Tree Classifier** trained on the 
        [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
        
        #### 📊 Dataset Information
        - **Source**: UCI Machine Learning Repository
        - **Samples**: 569 (357 Benign, 212 Malignant)
        - **Features**: 30 numeric features computed from digitized images of fine needle aspirates (FNA)
        
        #### 🔬 Features Explained
        The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image:
        
        | Feature | Description |
        |---------|-------------|
        | Radius | Mean of distances from center to points on the perimeter |
        | Texture | Standard deviation of gray-scale values |
        | Perimeter | Perimeter of the cell nucleus |
        | Area | Area of the cell nucleus |
        | Smoothness | Local variation in radius lengths |
        | Compactness | Perimeter² / Area - 1.0 |
        | Concavity | Severity of concave portions of the contour |
        | Concave Points | Number of concave portions of the contour |
        | Symmetry | Symmetry of the cell nucleus |
        | Fractal Dimension | "Coastline approximation" - 1 |
        
        #### 🧠 Model Details
        - **Algorithm**: Decision Tree Classifier
        - **Hyperparameters**: Tuned using individual exploration (see notebook 4)
        - **Criterion**: Entropy (Information Gain)
        - **Max Depth**: 3
        
        #### ⚠️ Disclaimer
        This tool is for **educational and demonstration purposes only**. 
        It should **NOT** be used for actual medical diagnosis. 
        Always consult qualified healthcare professionals for medical decisions.
        
        ---
        
        **Built with ❤️ by Ganesh Khetawat**
        
        [GitHub Repository](https://github.com/gkganesh12/Breast_cancer_Prediction_model)
        """)

if __name__ == "__main__":
    main()
