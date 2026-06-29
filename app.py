import io
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Telecom Churn SQL + Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "clean" / "telecom_churn_clean.csv"
TABLE_NAME = "telecom_churn"


def _normalize_binary(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        mapping = {
            "yes": 1,
            "true": 1,
            "1": 1,
            "churn": 1,
            "no": 0,
            "false": 0,
            "0": 0,
            "not churn": 0,
        }
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )
    return series.astype(int)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    if "churn" in df.columns:
        df["churn"] = _normalize_binary(df["churn"])

    return df


@st.cache_resource
def build_db(df: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
    return conn


def generate_sql_from_text(user_text: str) -> str:
    q = user_text.lower().strip()

    if q.startswith(("select", "with")):
        return user_text

    if "customers" in q and "plan" in q:
        return f"""
        SELECT
          international_plan,
          voice_mail_plan,
          COUNT(*) AS customers,
          ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
        FROM {TABLE_NAME}
        GROUP BY international_plan, voice_mail_plan
        ORDER BY customers DESC;
        """

    if "churn" in q and "area" in q:
        return f"""
        SELECT
          area_code,
          COUNT(*) AS customers,
          ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
        FROM {TABLE_NAME}
        GROUP BY area_code
        ORDER BY churn_rate_pct DESC;
        """

    if "customer service" in q or "support" in q:
        return f"""
        SELECT
          customer_service_calls,
          COUNT(*) AS customers,
          ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
        FROM {TABLE_NAME}
        GROUP BY customer_service_calls
        ORDER BY customer_service_calls;
        """

    if "intl" in q or "international" in q:
        return f"""
        SELECT
          international_plan,
          ROUND(AVG(total_intl_minutes), 2) AS avg_intl_minutes,
          ROUND(AVG(total_intl_charge), 2) AS avg_intl_charge,
          ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
        FROM {TABLE_NAME}
        GROUP BY international_plan;
        """

    return f"SELECT * FROM {TABLE_NAME} LIMIT 50;"


def execute_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    sql_clean = sql.strip().lower()
    if not sql_clean.startswith(("select", "with")):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")

    blocked = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
    if any(word in sql_clean for word in blocked):
        raise ValueError("Only read-only analytics queries are allowed.")

    return pd.read_sql_query(sql, conn)


@st.cache_resource
def train_model(df: pd.DataFrame):
    if "churn" not in df.columns:
        raise ValueError("The dataset must contain a 'churn' column for prediction.")

    X = df.drop(columns=["churn"])
    y = df["churn"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return model, auc, X.columns.tolist()


def render_visual_and_download(df: pd.DataFrame):
    if df.empty:
        st.info("No rows to visualize.")
        return

    with st.expander("View visualisation", expanded=True):
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in numeric_cols]

        if not numeric_cols:
            st.info("Need at least one numeric column to plot.")
            return

        default_x = cat_cols[0] if cat_cols else df.columns[0]
        default_y = numeric_cols[0]

        x_col = st.selectbox("X axis", df.columns.tolist(), index=df.columns.tolist().index(default_x))
        y_col = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index(default_y))
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter"])

        fig, ax = plt.subplots(figsize=(8, 4.8))

        if chart_type == "Bar":
            ax.bar(df[x_col].astype(str), df[y_col], color="#0f9d8a")
        elif chart_type == "Line":
            ax.plot(df[x_col].astype(str), df[y_col], marker="o", color="#1654a0")
        else:
            ax.scatter(df[x_col].astype(str), df[y_col], color="#0b6d60")

        ax.set_title(f"{y_col} by {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        png_bytes = io.BytesIO()
        fig.savefig(png_bytes, format="png", dpi=180, bbox_inches="tight")
        png_bytes.seek(0)

        st.download_button(
            label="Download visualisation (PNG)",
            data=png_bytes,
            file_name="telecom_visualisation.png",
            mime="image/png",
        )


def main():
    st.title("Telecom Churn: Chat SQL Analytics + Prediction")
    st.caption("Type a business query, run SQL-style analytics, visualize output, and predict churn in one app.")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    conn = build_db(df)

    try:
        model, auc, model_features = train_model(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Model AUC", f"{auc:.3f}")

    tabs = st.tabs(["Chat SQL Analytics", "Prediction"])

    with tabs[0]:
        st.subheader("Business query to SQL")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_prompt = st.chat_input("Example: show me customers by plan")
        if user_prompt:
            sql = generate_sql_from_text(user_prompt)
            try:
                result_df = execute_sql(conn, sql)
                st.session_state.last_result_df = result_df
                st.session_state.chat_history.append(
                    {"prompt": user_prompt, "sql": sql, "rows": len(result_df)}
                )
            except Exception as exc:
                st.error(f"Query failed: {exc}")

        if st.session_state.get("chat_history"):
            for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
                with st.container(border=True):
                    st.write(f"Query {idx}: {item['prompt']}")
                    st.code(item["sql"], language="sql")
                    st.caption(f"Rows returned: {item['rows']}")

        if "last_result_df" in st.session_state:
            st.subheader("Query Output")
            st.dataframe(st.session_state.last_result_df, use_container_width=True)

            csv_bytes = st.session_state.last_result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download output (CSV)",
                data=csv_bytes,
                file_name="query_output.csv",
                mime="text/csv",
            )

            render_visual_and_download(st.session_state.last_result_df)

    with tabs[1]:
        st.subheader("Single customer churn prediction")

        with st.form("predict_form"):
            input_data = {}
            for feature in model_features:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    f_min = float(df[feature].min())
                    f_max = float(df[feature].max())
                    f_mean = float(df[feature].mean())
                    input_data[feature] = st.slider(
                        feature,
                        min_value=f_min,
                        max_value=f_max,
                        value=f_mean,
                    )
                else:
                    options = sorted(df[feature].dropna().astype(str).unique().tolist())
                    input_data[feature] = st.selectbox(feature, options)

            submit = st.form_submit_button("Predict churn risk")

        if submit:
            input_df = pd.DataFrame([input_data])
            churn_proba = float(model.predict_proba(input_df)[0][1])
            label = "High risk" if churn_proba >= 0.5 else "Low risk"
            st.metric("Predicted churn probability", f"{churn_proba:.2%}")
            st.write(f"Risk label: **{label}**")

        if "last_result_df" in st.session_state and not st.session_state.last_result_df.empty:
            st.subheader("Batch score latest analytics output")
            batch_df = st.session_state.last_result_df.copy()

            missing_features = [c for c in model_features if c not in batch_df.columns]
            if missing_features:
                st.info(
                    "Latest query output does not include all model features. "
                    "Use SELECT * or include feature columns for batch scoring."
                )
            else:
                scored_df = batch_df.copy()
                scored_df["predicted_churn_probability"] = model.predict_proba(batch_df[model_features])[:, 1]
                scored_df["predicted_churn_label"] = (
                    scored_df["predicted_churn_probability"] >= 0.5
                ).map({True: "High risk", False: "Low risk"})

                st.dataframe(scored_df, use_container_width=True)
                st.download_button(
                    label="Download scored output (CSV)",
                    data=scored_df.to_csv(index=False).encode("utf-8"),
                    file_name="scored_output.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
