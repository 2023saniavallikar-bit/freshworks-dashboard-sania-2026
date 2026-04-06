import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Freshworks Workforce Productivity Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Freshworks: Non-Technical Skills and Productivity Dashboard")
st.caption(
    "Course: Big Data and Cloud Computing | Focus: Communication, Collaboration, "
    "and Workforce Productivity in cloud-based work environments"
)

DATA_PATH = "Cloud_HR_Productivity_Analytics_Dataset_v2.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "productivity_level" not in df.columns:
        df["productivity_level"] = pd.cut(
            df["productivity_score"],
            bins=[-1, 50, 75, 100],
            labels=["Low", "Medium", "High"],
        )
    return df


df = load_data(DATA_PATH)

with st.sidebar:
    st.header("Filters")
    departments = ["All"] + sorted(df["department"].dropna().unique().tolist())
    job_roles = ["All"] + sorted(df["job_role"].dropna().unique().tolist())
    platforms = ["All"] + sorted(df["cloud_platform_usage"].dropna().unique().tolist())

    selected_department = st.selectbox("Department", departments, index=0)
    selected_job_role = st.selectbox("Job Role", job_roles, index=0)
    selected_platform = st.selectbox("Cloud Platform", platforms, index=0)
    comm_range = st.slider("Communication Score", 0, 10, (0, 10))
    collab_range = st.slider("Collaboration Score", 0, 10, (0, 10))

filtered_df = df.copy()
if selected_department != "All":
    filtered_df = filtered_df[filtered_df["department"] == selected_department]
if selected_job_role != "All":
    filtered_df = filtered_df[filtered_df["job_role"] == selected_job_role]
if selected_platform != "All":
    filtered_df = filtered_df[filtered_df["cloud_platform_usage"] == selected_platform]

filtered_df = filtered_df[
    filtered_df["communication_score"].between(comm_range[0], comm_range[1])
    & filtered_df["collaboration_score"].between(collab_range[0], collab_range[1])
]

if filtered_df.empty:
    st.warning("No records match the selected filters. Please widen your filter range.")
    st.stop()

# KPI row
total_employees_full = int(df.shape[0])
total_employees_scope = int(filtered_df.shape[0])
avg_productivity = float(filtered_df["productivity_score"].mean())
high_share = (
    filtered_df["productivity_level"].eq("High").mean() * 100
    if "productivity_level" in filtered_df.columns
    else 0.0
)
attrition_high = (
    filtered_df["attrition_risk"].astype(str).str.lower().eq("high").mean() * 100
    if "attrition_risk" in filtered_df.columns
    else 0.0
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Employees (Dataset)", f"{total_employees_full:,}")
col2.metric("Employees in Current Filter", f"{total_employees_scope:,}")
col3.metric("Avg Productivity Score", f"{avg_productivity:.2f}")
col4.metric("High Productivity Share", f"{high_share:.1f}%")
col5.metric("High Attrition Risk Share", f"{attrition_high:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive View", "Skill Impact", "Department View", "Recommendations"]
)

with tab1:
    st.subheader("Executive Productivity Snapshot")

    prod_counts = (
        filtered_df["productivity_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
    )
    fig_prod = go.Figure(
        data=[
            go.Bar(
                x=prod_counts.index,
                y=prod_counts.values,
                text=prod_counts.values,
                textposition="outside",
                marker_color=["#ef4444", "#f59e0b", "#10b981"],
            )
        ]
    )
    fig_prod.update_layout(
        title="Employee Distribution by Productivity Level",
        xaxis_title="Productivity Level",
        yaxis_title="Employee Count",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_prod, use_container_width=True)

    st.markdown(
        "- Most employees are concentrated in Low/Medium bands, so targeted upskilling "
        "creates the biggest improvement opportunity.\n"
        "- A small increase in communication and collaboration can move a larger group "
        "into the high-productivity band."
    )

with tab2:
    st.subheader("Impact of Non-Technical Skills")

    scatter = px.scatter(
        filtered_df.sample(min(8000, len(filtered_df)), random_state=42),
        x="communication_score",
        y="collaboration_score",
        color="productivity_score",
        color_continuous_scale="Viridis",
        opacity=0.65,
        title="Communication + Collaboration vs Productivity",
        labels={
            "communication_score": "Communication Score",
            "collaboration_score": "Collaboration Score",
            "productivity_score": "Productivity Score",
        },
    )
    st.plotly_chart(scatter, use_container_width=True)

    corr_cols = [
        "communication_score",
        "collaboration_score",
        "leadership_score",
        "problem_solving_score",
        "productivity_score",
    ]
    corr = filtered_df[corr_cols].corr(numeric_only=True)
    heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap: Skills vs Productivity",
    )
    st.plotly_chart(heat, use_container_width=True)

    comm_trend = (
        filtered_df.groupby("communication_score", as_index=False)["productivity_score"]
        .mean()
        .sort_values("communication_score")
    )
    collab_trend = (
        filtered_df.groupby("collaboration_score", as_index=False)["productivity_score"]
        .mean()
        .sort_values("collaboration_score")
    )

    col_a, col_b = st.columns(2)
    with col_a:
        comm_fig = px.line(
            comm_trend,
            x="communication_score",
            y="productivity_score",
            markers=True,
            title="Communication Score vs Avg Productivity",
            labels={
                "communication_score": "Communication Score",
                "productivity_score": "Avg Productivity Score",
            },
        )
        st.plotly_chart(comm_fig, use_container_width=True)
    with col_b:
        collab_fig = px.line(
            collab_trend,
            x="collaboration_score",
            y="productivity_score",
            markers=True,
            title="Collaboration Score vs Avg Productivity",
            labels={
                "collaboration_score": "Collaboration Score",
                "productivity_score": "Avg Productivity Score",
            },
        )
        st.plotly_chart(collab_fig, use_container_width=True)

    skill_by_prod = (
        filtered_df.groupby("productivity_level", as_index=False)[
            ["communication_score", "collaboration_score", "leadership_score"]
        ]
        .mean()
        .melt(
            id_vars="productivity_level",
            var_name="skill",
            value_name="avg_score",
        )
    )
    skill_fig = px.bar(
        skill_by_prod,
        x="productivity_level",
        y="avg_score",
        color="skill",
        barmode="group",
        title="Average Skill Scores by Productivity Level",
        labels={"productivity_level": "Productivity Level", "avg_score": "Avg Skill Score"},
    )
    st.plotly_chart(skill_fig, use_container_width=True)

with tab3:
    st.subheader("Department Productivity and Work Patterns")

    dept_perf = (
        filtered_df.groupby("department", as_index=False)["productivity_score"]
        .mean()
        .sort_values("productivity_score", ascending=False)
    )
    dept_chart = px.bar(
        dept_perf,
        x="department",
        y="productivity_score",
        color="productivity_score",
        color_continuous_scale="Blues",
        text=dept_perf["productivity_score"].round(2),
        title="Average Productivity by Department",
    )
    dept_chart.update_traces(textposition="outside")
    st.plotly_chart(dept_chart, use_container_width=True)

    meeting_perf = (
        filtered_df.groupby("meetings_attended", as_index=False)["productivity_score"]
        .mean()
        .sort_values("meetings_attended")
    )
    meet_chart = px.line(
        meeting_perf,
        x="meetings_attended",
        y="productivity_score",
        markers=True,
        title="Meetings Attended vs Average Productivity",
        labels={
            "meetings_attended": "Meetings Attended",
            "productivity_score": "Avg Productivity Score",
        },
    )
    st.plotly_chart(meet_chart, use_container_width=True)

    top_10 = filtered_df.nlargest(10, "productivity_score")[
        [
            "employee_id",
            "department",
            "communication_score",
            "collaboration_score",
            "productivity_score",
        ]
    ]
    bottom_10 = filtered_df.nsmallest(10, "productivity_score")[
        [
            "employee_id",
            "department",
            "communication_score",
            "collaboration_score",
            "productivity_score",
        ]
    ]

    st.markdown("#### Top 10 and Bottom 10 Employees by Productivity")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(top_10, use_container_width=True, hide_index=True)
    with c2:
        st.dataframe(bottom_10, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Data-Backed Recommendations for Freshworks")
    st.markdown(
        "1. **Targeted Skill Uplift:** Prioritize communication and collaboration coaching "
        "for low-performing groups; these two skills show the strongest positive productivity relationship.\n"
        "2. **Department Action Plans:** Run focused interventions for departments below overall average "
        "productivity and review improvement every month.\n"
        "3. **High-Risk Retention Strategy:** Combine productivity and attrition-risk signals to identify "
        "employees needing manager support and career conversations.\n"
        "4. **Meeting Effectiveness Policy:** Use meetings-vs-productivity trend to define an optimal meeting band "
        "and reduce low-value meetings.\n"
        "5. **Replicate Top Performer Patterns:** Benchmark top 10 employees' communication/collaboration profiles "
        "and convert them into team-level best practices.\n"
        "6. **Business KPI Tracking:** Track movement from Low -> Medium -> High productivity monthly and link it "
        "to training ROI and delivery outcomes."
    )

st.divider()
st.caption(
    "How to read this dashboard: use filters to compare groups, identify weak skill "
    "profiles, and prioritize interventions with measurable productivity impact."
)
