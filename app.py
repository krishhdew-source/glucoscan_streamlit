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
def load_and_process_glucose_data(csv_file_path):
    """
    Load glucose data from CSV, which is already categorized.
    
    Args:
        csv_file_path: Path to the CSV file with glucose readings
        
    Returns:
        pandas DataFrame: Data with 'datetime' column added and sorted
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        if 'time' not in df.columns or 'date' not in df.columns or 'category' not in df.columns:
            st.error("CSV missing 'date', 'time', or 'category' columns")
            return None
        
        # Convert glucose_reading to numeric, handling errors
        df['glucose_reading'] = pd.to_numeric(df['glucose_reading'], errors='coerce')
        
        # Filter out rows with non-numeric glucose readings
        df = df.dropna(subset=['glucose_reading'])
        
        # Combine date and time into datetime column, handling potential errors
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'], 
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        df = df.dropna(subset=['datetime'])
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    except FileNotFoundError:
        st.error(f"CSV file not found at {csv_file_path}")
        return None
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None


def create_category_chart(df, category_name, color):
    """
    Create a chart for a specific glucose category.
    
    Args:
        df: DataFrame with all glucose data
        category_name: The category to plot (e.g., "Fasting")
        color: The color for the chart line
        
    Returns:
        plotly figure object
    """
    category_data = df[df['category'] == category_name].copy()
    
    fig = go.Figure()
    
    if not category_data.empty:
        fig.add_trace(go.Scatter(
            x=category_data['datetime'],
            y=category_data['glucose_reading'],
            mode='lines+markers',
            name=category_name,
            line=dict(color=color, width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor=f'rgba({",".join(str(c) for c in px.colors.hex_to_rgb(color))}, 0.2)'
        ))
        
        # Add average line
        avg_glucose = category_data['glucose_reading'].mean()
        fig.add_hline(
            y=avg_glucose,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_glucose:.1f}",
            annotation_position="bottom right"
        )
    
    fig.update_layout(
        title=f'{category_name} Readings',
        xaxis_title='Date & Time',
        yaxis_title='Glucose (mg/dL)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


# ===============================
# UI Layout
# ===============================
st.title("🩺 Glucose Dashboard")
st.write("Analyze glucose readings categorized by meal times.")

# CSV file path
csv_file = os.path.join(os.path.dirname(__file__), 'output_with_glucose copy.csv')

# Check for CSV file changes and auto-rerun
check_csv_changes(csv_file)

# Load data
if os.path.exists(csv_file):
    df = load_and_process_glucose_data(csv_file)
    
    if df is not None and not df.empty:
        # Define categories and their colors for consistent plotting
        category_config = {
            "Fasting": "#1f77b4",
            "Post-Breakfast": "#ff7f0e",
            "Pre-Dinner": "#2ca02c",
            "Post-Dinner": "#d62728",
            "Random": "#9467bd",
            "PPBS": "#e377c2",  # Added PPBS for backward compatibility
            "Uncategorized": "#8c564b"
        }
        
        # Get a sorted list of categories present in the data
        categories_in_data = sorted([cat for cat in df['category'].unique() if cat in category_config])

        # Display summary metrics
        st.divider()
        st.subheader("📊 Summary Statistics")
        
        # Use columns for metrics, dynamically create them
        metric_cols = st.columns(len(categories_in_data) + 1)
        with metric_cols[0]:
            st.metric("Total Readings", len(df))
        
        for i, category in enumerate(categories_in_data):
            with metric_cols[i+1]:
                count = len(df[df['category'] == category])
                st.metric(f"{category} Readings", count)

        st.divider()

        # Display charts and data for each category
        st.subheader("📈 Category Analysis")
        
        for i in range(0, len(categories_in_data), 2):
            col1, col2 = st.columns(2)
            
            # Display chart and data for the first category in the row
            with col1:
                category1 = categories_in_data[i]
                color1 = category_config.get(category1, "#000000")
                
                st.plotly_chart(create_category_chart(df, category1, color1), use_container_width=True)
                
                category_df1 = df[df['category'] == category1][['datetime', 'glucose_reading']]
                if not category_df1.empty:
                    st.dataframe(category_df1.style.format({'datetime': '{:%Y-%m-%d %H:%M:%S}'}), use_container_width=True)
                    st.write(f"**Average:** {category_df1['glucose_reading'].mean():.1f} mg/dL")
                    st.write(f"**Range:** {category_df1['glucose_reading'].min():.0f} - {category_df1['glucose_reading'].max():.0f} mg/dL")
                else:
                    st.info(f"No readings for {category1}")

            # Display chart and data for the second category in the row, if it exists
            if i + 1 < len(categories_in_data):
                with col2:
                    category2 = categories_in_data[i+1]
                    color2 = category_config.get(category2, "#000000")

                    st.plotly_chart(create_category_chart(df, category2, color2), use_container_width=True)
                    
                    category_df2 = df[df['category'] == category2][['datetime', 'glucose_reading']]
                    if not category_df2.empty:
                        st.dataframe(category_df2.style.format({'datetime': '{:%Y-%m-%d %H:%M:%S}'}), use_container_width=True)
                        st.write(f"**Average:** {category_df2['glucose_reading'].mean():.1f} mg/dL")
                        st.write(f"**Range:** {category_df2['glucose_reading'].min():.0f} - {category_df2['glucose_reading'].max():.0f} mg/dL")
                    else:
                        st.info(f"No readings for {category2}")
            st.divider()

        # Category comparison chart
        st.subheader("📊 Overall Category Comparison")
        category_stats = df.groupby('category')['glucose_reading'].agg(['mean', 'min', 'max', 'count']).round(1)
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=category_stats.index,
            y=category_stats['mean'],
            name='Average Reading',
            marker_color=[category_config.get(cat, "#000000") for cat in category_stats.index]
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
        if 'filename' not in df.columns:
            df['filename'] = 'N/A'
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

