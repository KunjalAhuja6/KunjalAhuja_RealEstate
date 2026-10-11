import sys
from pathlib import Path

# Ensure project root directory is on Python path so 'from src....' imports resolve
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import time

# Set page configuration
st.set_page_config(
    page_title="Real Estate Property Valuation Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports from backend modules
from src.rag.hybrid_retriever import hybrid_retriever
from src.agents.classification_agent import classification_agent
from src.agents.valuation_agent import valuation_agent
from src.agents.risk_agent import risk_agent
from src.agents.judge_agent import judge_agent
from src.agents.orchestrator import orchestrator
from src.services.ml_prediction import price_predictor
from src.services.feature_impact import feature_impact_service
from src.services.feedback_service import feedback_service, FeedbackSubmission
from src.services.evaluation_service import evaluation_runner
from src.rag.data_pipeline import load_cleaned_data

@st.cache_data
def get_neighborhood_list():
    df = load_cleaned_data()
    return sorted(df["Neighborhood"].unique().tolist())

neighborhoods = get_neighborhood_list()

# --- Custom Header ---
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 22px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0; color: white;">Real Estate Property Valuation Assistant</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1em;">
            Automated Property Appraisal System &bull; Multi-Index Retrieval &bull; Statistical Pricing Engine &bull; Risk Analysis &bull; Underwriting Audit
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
st.sidebar.header("Subject Property Characteristics")

selected_neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods, index=neighborhoods.index("North Ames") if "North Ames" in neighborhoods else 0)
gr_liv_area = st.sidebar.slider("Above-Ground Living Area (sq ft)", 500, 4500, 1800, step=50)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Full Bathrooms", 1, 4, 2)
year_built = st.sidebar.slider("Year Built", 1900, 2024, 1995)
year_remod = st.sidebar.slider("Year Remodeled / Modernized", 1950, 2026, 1995)
overall_qual = st.sidebar.slider("Overall Quality Grade (1-10)", 1, 10, 7)
overall_cond = st.sidebar.slider("Overall Condition Grade (1-10)", 1, 10, 6)
garage_cars = st.sidebar.slider("Garage Capacity (Cars)", 0, 4, 2)
total_bsmt_sf = st.sidebar.number_input("Basement Area (sq ft)", value=1000, step=100)
lot_area = st.sidebar.number_input("Lot Size (sq ft)", value=9500, step=500)
total_rooms = st.sidebar.number_input("Total Rooms Above Grade (Optional)", value=7, min_value=1, max_value=25, step=1)
garage_yr_blt = st.sidebar.number_input("Garage Year Built (Optional)", value=1995, min_value=1900, max_value=2026, step=1)

query_input = st.sidebar.text_input(
    "Property Inquiry / Search Terms",
    value=f"Estimate value of a {bedrooms} bedroom, {bathrooms} bath home in {selected_neighborhood} with {gr_liv_area} sq ft built in {year_built}",
)

target_features = {
    "neighborhood": selected_neighborhood,
    "gr_liv_area": float(gr_liv_area),
    "bedrooms": int(bedrooms),
    "bathrooms": float(bathrooms),
    "year_built": int(year_built),
    "year_remod_add": int(year_remod),
    "overall_qual": int(overall_qual),
    "overall_cond": int(overall_cond),
    "garage_cars": int(garage_cars),
    "total_bsmt_sf": float(total_bsmt_sf),
    "lot_area": float(lot_area),
    "total_rooms": int(total_rooms),
    "garage_yr_blt": int(garage_yr_blt),
}

# --- Main Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Property Valuation Dossier",
    "Market Comparables Explorer",
    "Valuation Modeling & Renovation Advisory",
    "Condition & Market Risk Assessment",
    "Market Feedback & Calibration",
    "Underwriting Decision Trail & Governance",
])

# ==============================================================================
# TAB 1: Property Valuation Dossier
# ==============================================================================
with tab1:
    st.subheader("Automated Property Valuation & Market Appraisal")
    st.caption("Synthesizes physical attributes, historical market comparables, statistical regression estimates, and independent underwriting review.")

    if st.button("Generate Comprehensive Valuation Report", type="primary", use_container_width=True):
        with st.spinner("Generating valuation analysis across appraisal modules..."):
            dossier = orchestrator.run_collaborative_valuation(
                property_features=target_features,
                query=query_input,
                top_k_comps=5,
            )
            st.session_state["latest_dossier"] = dossier

    if "latest_dossier" in st.session_state:
        dossier = st.session_state["latest_dossier"]

        # Top Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Estimated Fair Market Value", f"${dossier.reconciled_fair_market_value:,.0f}")
        with col2:
            st.metric("Recommended Valuation Range", f"${dossier.confidence_interval['low']:,.0f} – ${dossier.confidence_interval['high']:,.0f}")
        with col3:
            st.metric("Statistical Baseline Estimate", f"${dossier.ml_prediction['predicted_price']:,.0f}")
        with col4:
            st.metric("Comparable Sales Median", f"${dossier.basic_valuation.median_comparable_price:,.0f}")
        with col5:
            fb_info = feedback_service.get_feedback_summary()
            st.metric("Calibration Factor", f"{fb_info['calibration_factor']:.4f}")

        st.markdown("---")

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("### Executive Valuation Summary")
            st.info(dossier.final_executive_summary)

            st.markdown("### Market Classification")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.write(f"**Market Tier:** {dossier.classification.market_tier}")
                st.write(f"**Typology:** {dossier.classification.property_typology}")
            with c_col2:
                st.write(f"**Condition Grade:** {dossier.classification.condition_grade}")
                st.write(f"**Investment Profile:** {dossier.classification.investment_profile}")
            st.write(f"**Buyer Demographic:** {dossier.classification.target_demographic}")

        with col_right:
            st.markdown("### Automated Underwriting & Compliance Audit")
            verdict = dossier.judge_audit.verdict
            if verdict == "APPROVED":
                st.success(f"**Verdict:** {verdict} (Confidence: {dossier.judge_audit.confidence_score*100:.0f}%)")
            elif "FLAGGED" in verdict:
                st.error(f"**Verdict:** {verdict} (Confidence: {dossier.judge_audit.confidence_score*100:.0f}%)")
            else:
                st.warning(f"**Verdict:** {verdict} (Confidence: {dossier.judge_audit.confidence_score*100:.0f}%)")

            st.write(dossier.judge_audit.audit_commentary)

            st.markdown("**Appraisal Consistency & Compliance Scores:**")
            for crit, sc in dossier.judge_audit.rubric_scores.items():
                label = crit.replace('_', ' ').title().replace("Comp", "Comparable").replace("Ml Model", "Statistical Model")
                st.progress(sc, text=f"{label}: {sc*100:.0f}%")

            if dossier.judge_audit.flags:
                st.markdown("**Underwriting Notes & Flags:**")
                for fl in dossier.judge_audit.flags:
                    st.warning(f"• {fl}")
    else:
        st.info("Click 'Generate Comprehensive Valuation Report' above to evaluate the subject property.")

# ==============================================================================
# TAB 2: Market Comparables Explorer
# ==============================================================================
with tab2:
    st.subheader("Market Comparables Explorer & Multi-Factor Alignment")
    st.caption("Retrieves historical arm's-length transactions combining lexical keyword search, dense semantic indexing, and multi-factor physical comparability ranking.")

    c_s1, c_s2 = st.columns([3, 1])
    with c_s1:
        search_q = st.text_input("Enter Search Terms / Target Specs", value=query_input, key="tab2_q")
    with c_s2:
        top_k = st.selectbox("Comps to Retrieve", [3, 5, 10, 15], index=1)

    comps = hybrid_retriever.retrieve(
        query=search_q,
        top_k=top_k,
        target_property=target_features,
    )

    if comps:
        comp_rows = []
        for c in comps:
            d = c["data"]
            comp_rows.append({
                "Property ID": c["property_id"],
                "Neighborhood": d.get("Neighborhood"),
                "Prioritized Score": f"{c['prioritized_score']*100:.1f}%",
                "Comparability Score": f"{c['comparability_score']*100:.1f}%",
                "Sale Price": f"${d.get('SalePrice', 0):,}",
                "Living Area (sqft)": d.get("GrLivArea"),
                "Price / SqFt": f"${d.get('PricePerSqFt', 0):.2f}",
                "Quality": f"{d.get('OverallQual')}/10",
                "Year Built": d.get("YearBuilt"),
                "Beds / Baths": f"{d.get('BedroomAbvGr')} / {d.get('FullBath')}",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        st.markdown("### Detailed Comparable Evidence Breakdown")
        for i, c in enumerate(comps[:3]):
            with st.expander(f"Comparable #{i+1}: {c['property_id']} - {c['data'].get('Neighborhood')} (${c['data'].get('SalePrice', 0):,})"):
                st.write(f"**Searchable Document:** {c['text']}")
                st.json(c["comparability_details"])
    else:
        st.info("No comparable properties matched the query.")

# ==============================================================================
# TAB 3: Valuation Modeling & Renovation Advisory
# ==============================================================================
with tab3:
    st.subheader("Valuation Modeling & Renovation Advisory")
    st.caption("Statistical regression models evaluated on historical housing records with feature elasticity and renovation scenario modeling.")

    ml_res = price_predictor.predict(target_features)
    loc_res = feature_impact_service.get_location_impact(selected_neighborhood)
    feat_imp = feature_impact_service.get_feature_importances()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Statistical Estimate", f"${ml_res['predicted_price']:,.2f}")
    with col_m2:
        st.metric("90% Confidence Interval", f"${ml_res['lower_bound_90pct']:,.0f} – ${ml_res['upper_bound_90pct']:,.0f}")
    with col_m3:
        st.metric("Model Benchmark R²", f"{ml_res['model_r2_score']:.4f}")

    st.markdown("---")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("### Location Impact Analysis")
        st.write(f"**Neighborhood:** {loc_res.get('neighborhood', 'N/A')}")
        st.write(f"**Location Premium vs City Median:** `{loc_res.get('location_premium_pct', 0.0):+,.1f}%`")
        st.write(f"**Dollar Premium vs City:** `${loc_res.get('dollar_premium_vs_city', 0.0):+,.0f}`")
        st.write(f"**Neighborhood Median Price:** `${loc_res.get('median_neighborhood_price', 0.0):,.0f}`")
        st.write(f"**City-Wide Baseline Median:** `${loc_res.get('city_baseline_median', loc_res.get('city_median_price', 160000.0)):,.0f}`")

    with col_l2:
        st.markdown("### Top Feature Drivers")
        top_drivers = feat_imp["top_drivers"]
        df_imp = pd.DataFrame(list(top_drivers.items()), columns=["Feature", "Importance Weight"]).head(7)
        st.bar_chart(df_imp.set_index("Feature"))

    st.markdown("---")
    st.markdown("### What-If Renovation & Addition Simulator")
    st.caption("Simulate value added from home expansions, quality upgrades, or modernization.")

    c_w1, c_w2, c_w3, c_w4 = st.columns(4)
    with c_w1:
        sim_add_sqft = st.number_input("Add Living Area (sq ft)", value=250, step=50)
    with c_w2:
        sim_qual = st.selectbox("Upgrade Quality Grade", [0, 1, 2], index=1)
    with c_w3:
        sim_cars = st.selectbox("Add Garage Stall", [0, 1, 2], index=0)
    with c_w4:
        sim_remodel = st.checkbox("Complete Modern Remodel", value=True)

    if st.button("Run What-If Simulation"):
        mods = {
            "add_sqft": sim_add_sqft,
            "upgrade_quality": sim_qual,
            "add_garage_cars": sim_cars,
            "remodel": sim_remodel,
        }
        sim_res = feature_impact_service.simulate_what_if_scenario(target_features, mods)
        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            st.metric("Baseline Valuation", f"${sim_res['baseline_price']:,.0f}")
        with s_c2:
            st.metric("Simulated New Valuation", f"${sim_res['modified_price']:,.0f}")
        with s_c3:
            st.metric("Estimated Value Gained", f"+${sim_res['dollar_value_added']:,.0f}", f"+{sim_res['percentage_increase']}%")

# ==============================================================================
# TAB 4: Condition & Market Risk Assessment
# ==============================================================================
with tab4:
    st.subheader("Condition & Market Risk Assessment")
    st.caption("Quantifies age depreciation, condition discrepancies, neighborhood volatility, and layout liquidity risks.")

    risk_res = risk_agent.predict_risk(target_features)

    r_c1, r_c2 = st.columns([1, 2])
    with r_c1:
        st.metric("Overall Risk Score", f"{risk_res.overall_risk_score} / 100")
        if risk_res.risk_category == "Low Risk":
            st.success(f"**Category:** {risk_res.risk_category}")
        elif risk_res.risk_category == "Moderate Risk":
            st.info(f"**Category:** {risk_res.risk_category}")
        else:
            st.warning(f"**Category:** {risk_res.risk_category}")

    with r_c2:
        st.markdown("**Risk Dimensions:**")
        for dim, sc in risk_res.dimension_scores.items():
            st.progress(sc / 100.0, text=f"{dim.replace('_', ' ').title()}: {sc:.1f}/100")

    st.markdown("---")
    st.markdown("### Identified Property Vulnerabilities")
    for rf in risk_res.risk_factors:
        st.warning(f"• {rf}")

    st.markdown("### Recommended Mitigations")
    for rec in risk_res.mitigation_recommendations:
        st.info(f"• {rec}")

# ==============================================================================
# TAB 5: Market Feedback & Calibration
# ==============================================================================
with tab5:
    st.subheader("Market Feedback & Continuous Calibration")
    st.caption("Enables appraisers and real estate professionals to submit verified transaction outcomes to calibrate valuation bias.")

    fb_summary = feedback_service.get_feedback_summary()

    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        st.metric("Total Feedbacks Logged", fb_summary["total_feedback_count"])
    with f_c2:
        st.metric("Average Rating", f"{fb_summary['average_rating']:.1f} / 5.0")
    with f_c3:
        st.metric("Historical MAE", f"${fb_summary['mean_absolute_error_dollars']:,.0f}")
    with f_c4:
        st.metric("Calibration Bias Offset", f"{fb_summary['systematic_bias_pct']:+,.2f}%")

    st.markdown("---")
    st.markdown("### Submit Appraisal Outcome & Feedback")

    with st.form("feedback_form"):
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            rec_val_input = st.number_input("System Estimated Value ($)", value=210000.0, step=5000.0)
            actual_val_input = st.number_input("Actual Closing / Certified Appraiser Value ($)", value=215000.0, step=5000.0)
            rating = st.slider("Accuracy Rating (1 = Poor, 5 = Excellent)", 1, 5, 5)
        with col_fb2:
            role = st.selectbox("Professional Role", ["Certified Appraiser", "Licensed Broker / Realtor", "Mortgage Underwriter", "Investor / Buyer"])
            comments = st.text_area("Observations / Reasons for Price Discrepancy", "Transaction aligned closely with comps. Minor premium paid for lot orientation.")
            applied_cf_input = st.number_input("Applied Calibration Factor (from valuation)", value=float(fb_summary.get("calibration_factor", 1.0)), step=0.005, format="%.4f")

        submitted = st.form_submit_button("Submit Appraisal Feedback")
        if submitted:
            sub = FeedbackSubmission(
                property_id="USER-FEEDBACK",
                recommended_value=rec_val_input,
                actual_valuation=actual_val_input,
                applied_calibration_factor=applied_cf_input,
                user_rating=rating,
                appraiser_comments=comments,
                professional_role=role,
            )
            feedback_service.submit_feedback(sub)
            st.success("Feedback submitted and recorded to continuous learning repository.")
            st.rerun()

    st.markdown("### Recent Appraisal Records")
    if fb_summary["recent_records"]:
        st.dataframe(pd.DataFrame(fb_summary["recent_records"]), use_container_width=True)

# ==============================================================================
# TAB 6: Underwriting Decision Trail & Governance
# ==============================================================================
with tab6:
    st.subheader("Underwriting Decision Trail & Governance")
    st.caption("Regulatory compliance chain-of-custody log and automated model accuracy governance benchmarks.")

    st.markdown("### Section 1: Underwriting Decision Trail & Regulatory Chain-of-Custody")
    st.caption("Full audit log tracing intermediate actions and outputs across appraisal modules for defensibility.")

    if "latest_dossier" in st.session_state:
        dossier = st.session_state["latest_dossier"]
        trace_data = [
            {
                "Step": s.step_name,
                "Module": s.agent_name,
                "Action Taken": s.action_taken,
                "Outcome / Summary": s.output_summary,
            }
            for s in dossier.agent_trace
        ]
        st.table(pd.DataFrame(trace_data))
    else:
        st.info("Run a valuation in the 'Property Valuation Dossier' tab to generate and inspect the decision trail.")

    st.markdown("---")
    st.markdown("### Section 2: Valuation Model Accuracy & Governance Benchmark")
    st.caption("Standardized recommendation quality evaluation across representative market scenarios.")

    if st.button("Execute Governance Benchmark Suite", type="primary"):
        with st.spinner("Executing valuation benchmark framework across scenarios..."):
            bench = evaluation_runner.run_benchmark()

        eval_mode = bench.get("evaluation_mode", "offline_no_llm")
        st.markdown(f"**Benchmark Engine:** `{bench.get('evaluation_framework', 'DeepEval')}`")
        if eval_mode == "live_llm_judge":
            st.success(f"Mode: Live Underwriting Review (`{eval_mode}`) | Status: {bench['benchmark_status']}")
        else:
            st.info(f"Mode: Offline Statistical (`{eval_mode}`) | Status: {bench['benchmark_status']} — Review metrics require API credentials")

        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            pass_rate_str = f"{bench['pass_rate_percentage']}%" if bench.get('pass_rate_percentage') is not None else "N/A (Offline)"
            st.metric("Benchmark Pass Rate", pass_rate_str)
        with b_c2:
            faith = bench['mean_scores'].get('mean_faithfulness')
            st.metric("Factual Consistency", f"{faith*100:.1f}%" if faith is not None else "SKIPPED")
        with b_c3:
            rel = bench['mean_scores'].get('mean_answer_relevancy')
            st.metric("Query Relevancy", f"{rel*100:.1f}%" if rel is not None else "SKIPPED")

        b_c4, b_c5, b_c6 = st.columns(3)
        with b_c4:
            prec = bench['mean_scores'].get('mean_contextual_precision')
            st.metric("Contextual Precision", f"{prec*100:.1f}%" if prec is not None else "SKIPPED")
        with b_c5:
            rec = bench['mean_scores'].get('mean_contextual_recall')
            st.metric("Contextual Recall", f"{rec*100:.1f}%" if rec is not None else "SKIPPED")
        with b_c6:
            acc = bench['mean_scores'].get('mean_valuation_accuracy')
            st.metric("Valuation Accuracy", f"{acc*100:.1f}%" if acc is not None else "N/A")

        st.markdown("---")
        st.markdown("### Scenario Test Case Results")
        sc_data = []
        for s in bench["scenario_results"]:
            faith_val = s.get('metrics', {}).get('faithfulness')
            rel_val = s.get('metrics', {}).get('answer_relevancy')
            prec_val = s.get('metrics', {}).get('contextual_precision')
            status_val = "PASSED" if s.get("passed") is True else ("SKIPPED" if s.get("passed") is None else "FAILED")

            rec_val = s.get('reconciled_value')
            rec_val_str = f"${rec_val:,.0f}" if (rec_val is not None and rec_val > 0) else "N/A"

            sc_data.append({
                "Test ID": s["test_id"],
                "Scenario Name": s["test_name"],
                "Reconciled Value": rec_val_str,
                "Factual Consistency": f"{faith_val*100:.1f}%" if faith_val is not None else "SKIPPED",
                "Query Relevancy": f"{rel_val*100:.1f}%" if rel_val is not None else "SKIPPED",
                "Contextual Precision": f"{prec_val*100:.1f}%" if prec_val is not None else "SKIPPED",
                "Status": status_val,
            })
        st.table(pd.DataFrame(sc_data))
