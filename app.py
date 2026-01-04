import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import time

# ===============================
# Basic Page Setup
# ===============================
st.set_page_config(page_title="Glucose Dashboard", layout="wide")

# ===============================
# CSV File Change Detection
# ===============================
def check_csv_changes(csv_file_path):
    """
    Check if CSV file has been modified and trigger rerun if needed.
    """
    if not os.path.exists(csv_file_path):
        return
    
    # Get current file modification time
    current_mtime = os.path.getmtime(csv_file_path)
    
    # Initialize or check session state
    if 'csv_mtime' not in st.session_state:
        st.session_state.csv_mtime = current_mtime
    elif st.session_state.csv_mtime != current_mtime:
        # File has been modified, update session state and rerun
        st.session_state.csv_mtime = current_mtime
        st.rerun()
    
    # Set up auto-refresh every 5 seconds to detect changes
    if 'last_check' not in st.session_state:
        st.session_state.last_check = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_check > 5:
        st.session_state.last_check = current_time
        st.rerun()

# ===============================
# Data Processing Functions
# ===============================
def load_and_categorize_glucose_data(csv_file_path):
    """
    Load glucose data from CSV and categorize by time of day.
    
    Args:
        csv_file_path: Path to the CSV file with glucose readings
        
    Returns:
        pandas DataFrame: Data with 'category' and 'datetime' columns added
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        if 'time' not in df.columns or 'date' not in df.columns:
            st.error("CSV missing 'date' or 'time' columns")
            return None
        
        # Convert glucose_reading to numeric, handling errors
        df['glucose_reading'] = pd.to_numeric(df['glucose_reading'], errors='coerce')
        
        # Filter out rows with non-numeric glucose readings
        df = df.dropna(subset=['glucose_reading'])
        
        # Combine date and time into datetime column
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
        
        # Categorize based on time
        def categorize_time(time_str):
            try:
                time_obj = datetime.strptime(str(time_str).strip(), "%H:%M:%S").time()
                nine_am = datetime.strptime("09:30:00", "%H:%M:%S").time()
                return "Fasting" if time_obj < nine_am else "PPBS"
            except:
                return None
        
        df['category'] = df['time'].apply(categorize_time)
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    except FileNotFoundError:
        st.error(f"CSV file not found at {csv_file_path}")
        return None
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None


def create_glucose_chart(df):
    """
    Create an interactive chart showing glucose readings over time with categories.
    
    Args:
        df: DataFrame with glucose readings and category
        
    Returns:
        plotly figure object
    """
    fig = go.Figure()
    
    # Add Fasting readings
    fasting_data = df[df['category'] == 'Fasting']
    if len(fasting_data) > 0:
        fig.add_trace(go.Scatter(
            x=fasting_data['datetime'],
            y=fasting_data['glucose_reading'],
            mode='lines+markers',
            name='Fasting (Before 9 AM)',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
    
    # Add PPBS readings
    ppbs_data = df[df['category'] == 'PPBS']
    if len(ppbs_data) > 0:
        fig.add_trace(go.Scatter(
            x=ppbs_data['datetime'],
            y=ppbs_data['glucose_reading'],
            mode='lines+markers',
            name='PPBS (9 AM onwards)',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Glucose Readings Over Time',
        xaxis_title='Date & Time',
        yaxis_title='Glucose Reading (mg/dL)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_fasting_chart(df):
    """
    Create a chart showing only fasting readings (before 9 AM) over time.
    
    Args:
        df: DataFrame with glucose readings and category
        
    Returns:
        plotly figure object
    """
    fasting_data = df[df['category'] == 'Fasting'].copy()
    
    fig = go.Figure()
    
    if len(fasting_data) > 0:
        fig.add_trace(go.Scatter(
            x=fasting_data['datetime'],
            y=fasting_data['glucose_reading'],
            mode='lines+markers',
            name='Fasting',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10, symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        # Add average line
        avg_glucose = fasting_data['glucose_reading'].mean()
        fig.add_hline(
            y=avg_glucose,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_glucose:.1f} mg/dL",
            annotation_position="right"
        )
    
    fig.update_layout(
        title='Fasting Glucose Readings (Before 9 AM)',
        xaxis_title='Date & Time',
        yaxis_title='Glucose Reading (mg/dL)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_ppbs_chart(df):
    """
    Create a chart showing only PPBS readings (9 AM onwards) over time.
    
    Args:
        df: DataFrame with glucose readings and category
        
    Returns:
        plotly figure object
    """
    ppbs_data = df[df['category'] == 'PPBS'].copy()
    
    fig = go.Figure()
    
    if len(ppbs_data) > 0:
        fig.add_trace(go.Scatter(
            x=ppbs_data['datetime'],
            y=ppbs_data['glucose_reading'],
            mode='lines+markers',
            name='PPBS',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=10, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))
        
        # Add average line
        avg_glucose = ppbs_data['glucose_reading'].mean()
        fig.add_hline(
            y=avg_glucose,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_glucose:.1f} mg/dL",
            annotation_position="right"
        )
    
    fig.update_layout(
        title='PPBS Glucose Readings (9 AM onwards)',
        xaxis_title='Date & Time',
        yaxis_title='Glucose Reading (mg/dL)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig






# ===============================
# UI Layout
# ===============================
st.title("🩺 Glucose Dashboard")
st.write("Analyze glucose readings categorized by time of day (Fasting vs PPBS)")

# CSV file path
csv_file = os.path.join(os.path.dirname(__file__), 'output_with_glucose copy.csv')

# Check for CSV file changes and auto-rerun
check_csv_changes(csv_file)

# Load data
if os.path.exists(csv_file):
    df = load_and_categorize_glucose_data(csv_file)
    
    if df is not None and len(df) > 0:
        # Display statistics
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Readings", len(df))
        
        with col2:
            fasting_count = len(df[df['category'] == 'Fasting'])
            st.metric("Fasting Readings", fasting_count)
        
        with col3:
            ppbs_count = len(df[df['category'] == 'PPBS'])
            st.metric("PPBS Readings", ppbs_count)
        
        with col4:
            avg_glucose = df['glucose_reading'].mean()
            st.metric("Average Reading (mg/dL)", f"{avg_glucose:.1f}")
        
        st.divider()
        
        # Display separate category charts
        st.subheader("📈 Separate Category Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fasting = create_fasting_chart(df)
            st.plotly_chart(fig_fasting, use_container_width=True)
        
        with col2:
            fig_ppbs = create_ppbs_chart(df)
            st.plotly_chart(fig_ppbs, use_container_width=True)
        
        st.divider()
        
        # Display category statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Fasting Readings")
            fasting_df = df[df['category'] == 'Fasting'][['datetime', 'glucose_reading']]
            if len(fasting_df) > 0:
                st.dataframe(fasting_df, use_container_width=True)
                st.write(f"**Average:** {fasting_df['glucose_reading'].mean():.1f} mg/dL")
                st.write(f"**Range:** {fasting_df['glucose_reading'].min():.0f} - {fasting_df['glucose_reading'].max():.0f} mg/dL")
            else:
                st.info("No fasting readings available")
        
        with col2:
            st.subheader("📉 PPBS Readings")
            ppbs_df = df[df['category'] == 'PPBS'][['datetime', 'glucose_reading']]
            if len(ppbs_df) > 0:
                st.dataframe(ppbs_df, use_container_width=True)
                st.write(f"**Average:** {ppbs_df['glucose_reading'].mean():.1f} mg/dL")
                st.write(f"**Range:** {ppbs_df['glucose_reading'].min():.0f} - {ppbs_df['glucose_reading'].max():.0f} mg/dL")
            else:
                st.info("No PPBS readings available")
        
        st.divider()
        
        # Category comparison chart
        st.subheader("📊 Category Comparison")
        category_stats = df.groupby('category')['glucose_reading'].agg(['mean', 'min', 'max', 'count']).round(1)
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=category_stats.index,
            y=category_stats['mean'],
            name='Average Reading',
            marker=dict(color=['#1f77b4', '#ff7f0e'])
        ))
        
        fig_bar.update_layout(
            title='Average Glucose Reading by Category',
            xaxis_title='Category',
            yaxis_title='Glucose (mg/dL)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(category_stats, use_container_width=True)
        
        st.divider()
        
        # Download processed data
        csv_download = df[['date', 'time', 'glucose_reading', 'category', 'filename']].to_csv(index=False)
        st.download_button(
            label="Download Categorized Data (CSV)",
            data=csv_download,
            file_name="glucose_readings_categorized.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No valid glucose readings found in the CSV file")

else:
    st.error(f"CSV file not found at:\n{csv_file}\n\nPlease ensure the file exists.")
