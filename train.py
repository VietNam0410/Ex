import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import mlflow
import numpy as np
import time
import os

def train_models(X, y):
    st.header("Huấn Luyện Mô Hình Hình Học")  # Tiêu đề cho huấn luyện
    
    # Đặt đường dẫn lưu trữ MLflow offline (local)
    mlflow.set_tracking_uri("file:./mlruns")  # Lưu log vào thư mục 'mlruns'
    
    # Chuyển ảnh 28x28 thành vector 784 chiều
    X_flat = X.reshape(X.shape[0], -1)  # Từ (10000, 28, 28) thành (10000, 784)
    
    # Tỷ lệ chia tập dữ liệu
    st.subheader("Chia Tập Dữ Liệu")  # Tiêu đề cho phần chia dữ liệu
    test_size = st.slider("Tỷ lệ tập test (0.1-0.5)", 0.1, 0.5, 0.2, step=0.05)  # Chọn tỷ lệ test trước
    remaining_ratio = 1 - test_size  # Phần còn lại sau khi lấy tập test
    train_size = st.slider(
        f"Tỷ lệ tập train (từ phần còn lại {remaining_ratio:.2f})", 
        0.1, 0.9, 0.5, step=0.05
    ) * remaining_ratio  # Tỷ lệ train dựa trên phần còn lại
    val_size = 1 - test_size - train_size  # Tỷ lệ validation là phần còn lại
    
    # Hiển thị tỷ lệ các tập
    st.write(f"Tỷ lệ tập train: {train_size:.2f} ({train_size*100:.1f}%)")
    st.write(f"Tỷ lệ tập test: {test_size:.2f} ({test_size*100:.1f}%)")
    st.write(f"Tỷ lệ tập validation: {val_size:.2f} ({val_size*100:.1f}%)")
    
    # Kiểm tra tổng tỷ lệ
    total_ratio = train_size + test_size + val_size
    if not np.isclose(total_ratio, 1.0, rtol=1e-5):
        st.error(f"Tổng tỷ lệ không bằng 100%! Tổng: {total_ratio:.2f}")
        return
    
    # Chia tập test trước, sau đó chia train và validation
    X_temp, X_test, y_temp, y_test = train_test_split(X_flat, y, test_size=test_size, random_state=42)
    train_ratio = train_size / (train_size + val_size)  # Tỷ lệ train từ phần còn lại
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, train_size=train_ratio, random_state=42)
    
    # Hiển thị kích thước các tập để kiểm tra
    st.write(f"Kích thước tập train: {X_train.shape[0]} mẫu ({(X_train.shape[0]/X.shape[0]*100):.1f}%)")
    st.write(f"Kích thước tập validation: {X_val.shape[0]} mẫu ({(X_val.shape[0]/X.shape[0]*100):.1f}%)")
    st.write(f"Kích thước tập test: {X_test.shape[0]} mẫu ({(X_test.shape[0]/X.shape[0]*100):.1f}%)")
    
    # Kiểm tra tổng số mẫu
    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    if total_samples != X.shape[0]:
        st.error(f"Tổng số mẫu không khớp! Tổng: {total_samples}, Kỳ vọng: {X.shape[0]}")
        return
    else:
        st.success("Dữ liệu đã được chia đủ thành các tập train/validation/test!")
    
    # Chọn mô hình và tham số
    st.subheader("Cấu Hình Mô Hình")  # Tiêu đề cho phần chọn mô hình
    model_type = st.selectbox("Chọn mô hình", ["Hồi quy Logistic", "Cây Quyết định"])  # Bắt buộc chọn
    run_name = st.text_input("Nhập tên Run", f"{model_type}_HinhHoc_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Cross-validation settings
    st.subheader("Cấu Hình Cross-Validation")  # Tiêu đề cho phần cross-validation
    use_cv = st.checkbox("Sử dụng Cross-Validation", value=False)  # Tùy chọn sử dụng CV
    k_folds = 5  # Mặc định 5 folds
    if use_cv:
        k_folds = st.number_input("Số folds (k) cho Cross-Validation", 2, 10, 5)  # Người dùng nhập số folds
    
    # Cấu hình tham số mô hình
    if model_type == "Hồi quy Logistic":
        max_iter = st.number_input("Số lần lặp tối đa", min_value=1, value=100)  # Tham số tự do
        model = LogisticRegression(max_iter=max_iter)  # Khởi tạo mô hình
        
        # Tham số để log vào MLflow
        params = {
            "max_iter": max_iter
        }
        
    else:  # Cây Quyết định
        max_depth = st.number_input("Độ sâu tối đa", 1, 100, 10)  # Tham số tự do
        criterion = st.selectbox("Tiêu chí", ["gini", "entropy"])  # Bắt buộc chọn
        min_samples_split = st.number_input("Số mẫu tối thiểu để chia", 2, 20, 2)  # Tham số tự do
        model = DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split
        )  # Khởi tạo mô hình
        
        # Tham số để log vào MLflow
        params = {
            "max_depth": max_depth,
            "criterion": criterion,
            "min_samples_split": min_samples_split
        }
    
    # Cấu hình cổng cho MLflow UI
    st.subheader("Cấu Hình MLflow UI")  # Tiêu đề cho phần cấu hình MLflow
    mlflow_port = st.number_input("Cổng cho MLflow UI", 5000, 65535, 5000)  # Chọn cổng
    
    # Huấn luyện mô hình với thanh tiến trình
    if st.button("Huấn luyện mô hình"):
        progress_bar = st.progress(0)  # Tạo thanh tiến trình
        status_text = st.empty()  # Tạo placeholder để hiển thị trạng thái
        
        with mlflow.start_run(run_name=run_name) as run:  # Bắt đầu ghi log với MLflow
            # Huấn luyện mô hình
            status_text.text("Đang huấn luyện mô hình...")
            total_steps = 100 if not use_cv else 100 + k_folds * 10  # Tổng số bước
            step = 0
            
            # Huấn luyện trên tập train
            model.fit(X_train, y_train)
            for i in range(80):  # 80% tiến trình cho huấn luyện chính
                time.sleep(0.01)
                step += 1
                progress_bar.progress(int((step / total_steps) * 100))
            
            train_score = model.score(X_train, y_train)  # Điểm trên tập train
            val_score = model.score(X_val, y_val)  # Điểm trên tập validation
            
            # Cross-validation nếu được chọn
            cv_scores = None
            if use_cv:
                status_text.text(f"Đang thực hiện Cross-Validation với {k_folds} folds...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='accuracy')
                for i in range(k_folds * 10):  # 20% tiến trình cho CV
                    time.sleep(0.01)
                    step += 1
                    progress_bar.progress(int((step / total_steps) * 100))
            
            # Hoàn tất
            status_text.text("Huấn luyện hoàn tất!")
            progress_bar.progress(100)
            
            # Ghi log tham số và kết quả vào MLflow
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("use_cv", use_cv)
            if use_cv:
                mlflow.log_param("k_folds", k_folds)
            for param_name, param_value in params.items():  # Ghi log tất cả tham số mô hình
                mlflow.log_param(param_name, param_value)
            mlflow.log_metric("train_score", train_score)
            mlflow.log_metric("val_score", val_score)
            if use_cv:
                mlflow.log_metric("cv_mean_score", cv_scores.mean())
                mlflow.log_metric("cv_std_score", cv_scores.std())
            mlflow.sklearn.log_model(model, "model")  # Lưu mô hình
            
            # Hiển thị kết quả
            st.write(f"Điểm trên tập train: {train_score:.4f}")
            st.write(f"Điểm trên tập validation: {val_score:.4f}")
            if use_cv:
                st.write(f"Điểm Cross-Validation trung bình: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Hiển thị đường dẫn MLflow local (bỏ bớt thông báo)
            st.subheader("Kết Quả MLflow")
            mlruns_path = os.path.abspath("mlruns")  # Lấy đường dẫn tuyệt đối của thư mục mlruns
            st.write(f"Log đã được lưu tại: `{mlruns_path}`")
            st.code(f"mlflow ui --port {mlflow_port}")
            
            st.session_state[f"{model_type}_model"] = model  # Lưu mô hình vào session
            st.success("Huấn luyện mô hình thành công!")