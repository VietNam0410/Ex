import streamlit as st
import mlflow

def show_mlflow_logs():
    st.header("Theo Dõi Thử Nghiệm Hình Học MLflow")  # Tiêu đề cho theo dõi
    
    # Đặt đường dẫn lưu trữ MLflow offline (local)
    mlflow.set_tracking_uri("file:./mlruns")  # Lưu log vào thư mục 'mlruns'
    
    # Lấy tất cả các run đã lưu
    runs = mlflow.search_runs()  # Truy vấn các run từ MLflow
    
    if runs.empty:
        st.write("Không tìm thấy run nào!")  # Thông báo nếu không có dữ liệu
        return
        
    st.write("Các Run Trước Đó:")  # Hiển thị danh sách run
    st.dataframe(runs)  # Hiển thị toàn bộ thông tin của các run
    
    # Tạo danh sách run_name để người dùng chọn
    run_names = runs['tags.mlflow.runName'].tolist()
    run_ids = runs['run_id'].tolist()  # Lưu run_id để ánh xạ
    
    # Chọn run để xem chi tiết (dùng run_name thay vì run_id)
    selected_run_name = st.selectbox("Chọn run để xem chi tiết", run_names)
    if selected_run_name:
        # Lấy run_id tương ứng với run_name đã chọn
        selected_index = run_names.index(selected_run_name)
        selected_run_id = run_ids[selected_index]
        
        # Lấy thông tin chi tiết của run
        run = mlflow.get_run(selected_run_id)
        
        # Hiển thị run_name thay vì run_id
        st.subheader(f"Chi Tiết Run: {selected_run_name}")
        
        # Hiển thị tham số
        st.subheader("Tham Số")  # Hiển thị tham số
        st.write(run.data.params)
        
        # Hiển thị số liệu
        st.subheader("Số Liệu")  # Hiển thị số liệu
        st.write(run.data.metrics)