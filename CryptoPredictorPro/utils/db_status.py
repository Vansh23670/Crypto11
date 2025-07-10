import streamlit as st
from utils.database import DatabaseManager

def show_database_status():
    """Display database connection status in the sidebar"""
    try:
        db = DatabaseManager()
        if db.check_connection():
            st.sidebar.success("🟢 Database Connected")
        else:
            st.sidebar.error("🔴 Database Disconnected")
    except Exception as e:
        st.sidebar.error("🔴 Database Error")
        st.sidebar.caption(f"Error: {str(e)}")

def get_database_info():
    """Get database connection information"""
    try:
        db = DatabaseManager()
        if db.check_connection():
            return {
                'status': 'connected',
                'message': 'PostgreSQL database is running and connected'
            }
        else:
            return {
                'status': 'disconnected',
                'message': 'Database connection failed'
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Database error: {str(e)}'
        }