import streamlit as st
from process import load_and_process_data
from train import train_models
from demo import predict_demo
from mlflow_tracking import show_mlflow_logs

def main():
    st.title("Ứng Dụng Machine Learning Hình Học")  # Tiêu đề ứng dụng
    
    # Tạo các tab giao diện
    tab1, tab2, tab3, tab4 = st.tabs([
        "Xử Lý Dữ Liệu",      # Tab xử lý dữ liệu
        "Huấn Luyện Mô Hình",  # Tab huấn luyện mô hình
        "Dự Đoán Demo",        # Tab demo dự đoán
        "Theo Dõi MLflow"      # Tab theo dõi MLflow
    ])
    
    # Tab 1: Xử lý dữ liệu
    with tab1:
        data = load_and_process_data()  # Gọi hàm tải và minh họa dữ liệu
    
    # Tab 2: Huấn luyện mô hình
    with tab2:
        if 'X' in st.session_state and 'y' in st.session_state:
            train_models(st.session_state['X'], st.session_state['y'])  # Gọi hàm huấn luyện
        else:
            st.warning("Vui lòng tải dữ liệu hình học trước!")  # Thông báo nếu chưa có dữ liệu
    
    # Tab 3: Demo dự đoán
    with tab3:
        predict_demo()  # Gọi hàm dự đoán
    
    # Tab 4: Theo dõi MLflow
    with tab4:
        show_mlflow_logs()  # Gọi hàm hiển thị log MLflow

if __name__ == "__main__":
    main()  # Chạy hàm chính