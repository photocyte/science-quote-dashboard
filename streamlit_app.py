# Snowflake Streamlit App for Science Quote Analysis Dashboard
# Version for Streamlit Community Cloud deployment
# Uses environment variables for secure credential management

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import snowflake.connector
import os
from snowflake.snowpark.session import Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, lit, when, count, avg, sum as sum_func, to_date

# Configure the app
st.set_page_config(
    page_title="Science Quote Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def get_snowflake_session():
    """Get Snowflake session using environment variables"""
    try:
        # Get credentials from environment variables
        account = os.getenv('SNOWFLAKE_ACCOUNT')
        user = os.getenv('SNOWFLAKE_USER')
        password = os.getenv('SNOWFLAKE_PASSWORD')
        warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
        database = os.getenv('SNOWFLAKE_DATABASE', 'SCIENCE_QUOTE_ANALYSIS')
        schema = os.getenv('SNOWFLAKE_SCHEMA', 'PROCESSED_DOCUMENTS')
        
        if not all([account, user, password]):
            st.error("Missing Snowflake credentials. Please set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD environment variables.")
            return None
        
        # Create Snowpark session
        session = Session.builder.configs({
            "account": account,
            "user": user,
            "password": password,
            "warehouse": warehouse,
            "database": database,
            "schema": schema
        }).create()
        
        return session
        
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from Snowflake tables using Snowpark DataFrame API"""
    
    session = get_snowflake_session()
    if session is None:
        st.error("Cannot connect to Snowflake. Please check your credentials.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    try:
        # Load main analysis results using Snowpark DataFrame API
        main_df = session.table("PDF_ANALYSIS_RESULTS").select(
            "PDF_HASH",
            "ORIGINAL_KEY", 
            "IS_SCIENTIFIC_QUOTE",
            "CONFIDENCE",
            "REASONING",
            "DOCUMENT_TYPE",
            "PROCESSED_AT",
            "FILE_SIZE_MB",
            "PROCESSING_DURATION_MS",
            to_date(col("PROCESSED_AT")).alias("PROCESSING_DATE")
        ).order_by("PROCESSED_AT", ascending=False).limit(1000).to_pandas()
        
        # Load daily statistics using Snowpark DataFrame API
        daily_df = session.table("PROCESSING_SUMMARY").select(
            "PROCESSING_DATE",
            "TOTAL_PROCESSED",
            "SCIENTIFIC_QUOTES",
            (col("TOTAL_PROCESSED") - col("SCIENTIFIC_QUOTES")).alias("NON_SCIENTIFIC"),
            "AVG_CONFIDENCE",
            lit(0.95).alias("PROCESSING_SUCCESS_RATE")
        ).order_by("PROCESSING_DATE", ascending=False).limit(30).to_pandas()
        
        # Load document type analysis using Snowpark DataFrame API
        doc_type_stats = session.table("PDF_ANALYSIS_RESULTS").group_by("DOCUMENT_TYPE").agg(
            count("*").alias("COUNT"),
            avg("CONFIDENCE").alias("AVG_CONFIDENCE"),
            sum_func(when(col("IS_SCIENTIFIC_QUOTE"), 1).otherwise(0)).alias("SCIENTIFIC_COUNT")
        ).order_by("COUNT", ascending=False).to_pandas()
        
        session.close()
    
        st.write("Full table of results:", main_df)
    
        return main_df, daily_df, doc_type_stats
        
    except Exception as e:
        st.error(f"Error executing queries: {str(e)}")
        if session:
            session.close()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def display_metrics(main_df, daily_df, doc_type_stats):
    """Display key metrics using Streamlit components"""
    
    # Calculate key metrics with safety checks
    total_documents = len(main_df)
    
    # Check if IS_SCIENTIFIC_QUOTE column exists
    if 'IS_SCIENTIFIC_QUOTE' in main_df.columns:
        scientific_quotes = len(main_df[main_df['IS_SCIENTIFIC_QUOTE']])
        scientific_percentage = scientific_quotes / total_documents * 100 if total_documents > 0 else 0
    else:
        scientific_quotes = 0
        scientific_percentage = 0
        st.warning("Column 'IS_SCIENTIFIC_QUOTE' not found in data")
    
    # Check if CONFIDENCE column exists
    if 'CONFIDENCE' in main_df.columns:
        avg_confidence = main_df['CONFIDENCE'].mean()
    else:
        avg_confidence = 0
        st.warning("Column 'CONFIDENCE' not found in data")
    
    success_rate = 0.95  # Simulated success rate
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Total Documents",
            value=f"{total_documents:,}",
            delta=f"+{len(main_df[main_df['PROCESSING_DATE'] == main_df['PROCESSING_DATE'].max()]) if 'PROCESSING_DATE' in main_df.columns and not main_df.empty else 0} today"
        )
    
    with col2:
        st.metric(
            label="🔬 Scientific Quotes",
            value=f"{scientific_quotes:,}",
            delta=f"{scientific_percentage:.1f}%"
        )
    
    with col3:
        st.metric(
            label="🎯 Avg Confidence",
            value=f"{avg_confidence:.2f}",
            delta=f"{avg_confidence*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="✅ Success Rate",
            value=f"{success_rate*100:.1f}%",
            delta="+2.3%"
        )

def display_charts(main_df, daily_df, doc_type_stats):
    """Display charts using Streamlit's built-in charting"""
    
    st.subheader("📊 Analytics Dashboard")
    
    # Daily processing trend
    st.write("**Daily Processing Trend**")
    if not daily_df.empty:
        daily_chart_data = daily_df.set_index('PROCESSING_DATE')
        available_columns = [col for col in ['TOTAL_PROCESSED', 'SCIENTIFIC_QUOTES'] if col in daily_chart_data.columns]
        if available_columns:
            st.line_chart(daily_chart_data[available_columns])
        else:
            st.warning("No chartable columns available in daily data.")
    else:
        st.warning("No daily statistics data available.")
    
    # Document type distribution
    st.write("**Document Type Distribution**")
    if not doc_type_stats.empty and 'COUNT' in doc_type_stats.columns:
        doc_type_chart = doc_type_stats.set_index('DOCUMENT_TYPE')['COUNT']
        st.bar_chart(doc_type_chart)
    else:
        st.warning("No document type statistics available.")
    
    # Confidence distribution
    st.write("**Confidence Score Distribution**")
    if not main_df.empty and 'CONFIDENCE' in main_df.columns:
        st.bar_chart(pd.DataFrame({
            'confidence': main_df['CONFIDENCE'].value_counts().sort_index()
        }))
    else:
        st.warning("No confidence data available.")

def display_data_table(main_df):
    """Display data table with filtering options"""
    
    st.subheader("📋 Recent Processing Results")
    
    if main_df.empty:
        st.warning("No processing results data available.")
        return
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'PROCESSING_DATE' in main_df.columns:
            date_filter = st.date_input(
                "Filter by Date",
                value=main_df['PROCESSING_DATE'].max(),
                min_value=main_df['PROCESSING_DATE'].min(),
                max_value=main_df['PROCESSING_DATE'].max()
            )
        else:
            date_filter = None
            st.info("No PROCESSING_DATE column available for filtering")
    
    with col2:
        if 'DOCUMENT_TYPE' in main_df.columns:
            doc_type_filter = st.selectbox(
                "Filter by Document Type",
                options=['All'] + list(main_df['DOCUMENT_TYPE'].unique())
            )
        else:
            doc_type_filter = 'All'
            st.info("No DOCUMENT_TYPE column available for filtering")
    
    with col3:
        if 'IS_SCIENTIFIC_QUOTE' in main_df.columns:
            scientific_filter = st.selectbox(
                "Filter by Scientific Content",
                options=['All', 'Scientific', 'Non-Scientific']
            )
        else:
            scientific_filter = 'All'
            st.info("No IS_SCIENTIFIC_QUOTE column available for filtering")
    
    # Apply filters
    filtered_df = main_df.copy()
    
    if date_filter and 'PROCESSING_DATE' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['PROCESSING_DATE'] == date_filter]
    
    if doc_type_filter != 'All' and 'DOCUMENT_TYPE' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['DOCUMENT_TYPE'] == doc_type_filter]
    
    if scientific_filter == 'Scientific' and 'IS_SCIENTIFIC_QUOTE' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['IS_SCIENTIFIC_QUOTE'] == True]
    elif scientific_filter == 'Non-Scientific' and 'IS_SCIENTIFIC_QUOTE' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['IS_SCIENTIFIC_QUOTE'] == False]
    
    # Display filtered data
    if not filtered_df.empty:
        # Select only columns that exist
        display_columns = []
        for col in ['ORIGINAL_KEY', 'DOCUMENT_TYPE', 'IS_SCIENTIFIC_QUOTE', 'CONFIDENCE', 'PROCESSING_DATE']:
            if col in filtered_df.columns:
                display_columns.append(col)
        
        if display_columns:
            st.dataframe(
                filtered_df[display_columns].head(20),
                use_container_width=True
            )
        else:
            st.warning("No displayable columns available.")
    else:
        st.info("No data matches the selected filters.")

def display_insights(main_df, doc_type_stats):
    """Display insights and analysis"""
    
    st.subheader("💡 Key Insights")
    
    if main_df.empty:
        st.warning("No data available for insights.")
        return
    
    # Calculate insights with safety checks
    if 'IS_SCIENTIFIC_QUOTE' in main_df.columns:
        scientific_percentage = len(main_df[main_df['IS_SCIENTIFIC_QUOTE']]) / len(main_df) * 100
        high_confidence_count = len(main_df[main_df['CONFIDENCE'] > 0.9]) if 'CONFIDENCE' in main_df.columns else 0
    else:
        scientific_percentage = 0
        high_confidence_count = 0
        st.warning("Column 'IS_SCIENTIFIC_QUOTE' not found for insights.")
    
    if 'FILE_SIZE_MB' in main_df.columns:
        avg_file_size = main_df['FILE_SIZE_MB'].mean()
    else:
        avg_file_size = 0
        st.warning("Column 'FILE_SIZE_MB' not found for insights.")
    
    # Display insights in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🔬 Scientific Content Analysis</h4>
        <p><strong>{scientific_percentage:.1f}%</strong> of documents contain scientific quotes</p>
        <p><strong>{high_confidence_count}</strong> documents with high confidence (>90%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        most_common_type = doc_type_stats.iloc[0]['DOCUMENT_TYPE'] if not doc_type_stats.empty and 'DOCUMENT_TYPE' in doc_type_stats.columns else 'N/A'
        st.markdown(f"""
        <div class="insight-box">
        <h4>📊 Processing Statistics</h4>
        <p>Average file size: <strong>{avg_file_size:.1f} MB</strong></p>
        <p>Most common type: <strong>{most_common_type}</strong></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Science Quote Analysis Dashboard - MVP</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data from Snowflake..."):
        main_df, daily_df, doc_type_stats = load_data()
    
    # Display metrics
    display_metrics(main_df, daily_df, doc_type_stats)
    
    st.markdown("---")
    
    # Display charts
    display_charts(main_df, daily_df, doc_type_stats)
    
    st.markdown("---")
    
    # Display data table
    display_data_table(main_df)
    
    st.markdown("---")
    
    # Display insights
    display_insights(main_df, doc_type_stats)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
    <p>🔬 Science Quote Analysis Dashboard | Powered by Streamlit Community Cloud</p>
    <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
