import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mlflow
import mlflow.sklearn
from datetime import datetime

def predict_demo():
    st.header("Dự Đoán Hình Học")  # Tiêu đề cho demo
    
    # Khởi tạo lịch sử dự đoán trong session_state nếu chưa có
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    # Đặt tracking URI cho MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Lấy danh sách các run đã log trong MLflow
    runs = mlflow.search_runs()
    if runs.empty:
        st.warning("Không tìm thấy mô hình nào đã log trong MLflow! Vui lòng huấn luyện mô hình trước.")
        return
    
    # Tạo danh sách các run để người dùng chọn (chỉ hiển thị run_name và thời gian log)
    run_options = []
    run_ids = []
    for _, run in runs.iterrows():
        run_name = run['tags.mlflow.runName']
        start_time = run['start_time'].strftime("%Y-%m-%d %H:%M:%S")  # Định dạng thời gian
        run_options.append(f"{run_name} (Log tại: {start_time})")
        run_ids.append(run['run_id'])
    
    # Chọn mô hình
    selected_run = st.selectbox("Chọn mô hình đã log để dự đoán", run_options)
    
    # Lấy run_id từ lựa chọn
    selected_index = run_options.index(selected_run)
    selected_run_id = run_ids[selected_index]
    
    # Tải mô hình từ MLflow
    try:
        model_uri = f"runs:/{selected_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        st.error(f"Không thể tải mô hình từ MLflow: {str(e)}")
        return
    
    # Tải ảnh từ người dùng
    st.subheader("Tải Ảnh Lên")  # Tiêu đề cho phần tải ảnh
    uploaded_file = st.file_uploader("Tải ảnh 28x28 (grayscale)", type=['png', 'jpg', 'jpeg'], key="file_uploader")
    
    if uploaded_file is not None:
        # Đọc và xử lý ảnh
        image = Image.open(uploaded_file).convert('L')  # Chuyển thành grayscale
        image = image.resize((28, 28))  # Resize về 28x28
        image_array = np.array(image)  # Chuyển thành numpy array
        
        # Kiểm tra kích thước ảnh
        if image_array.shape != (28, 28):
            st.error("Ảnh phải có kích thước 28x28!")
            return
        
        # Hiển thị ảnh đã tải lên
        st.write("Ảnh đã tải lên:")
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
        # Nút để thực hiện dự đoán
        if st.button("Dự Đoán"):
            # Chuyển ảnh thành vector 784 chiều để dự đoán
            sample = image_array.reshape(1, -1)  # Từ (28, 28) thành (1, 784)
            
            # Dự đoán
            shape_names = ['square', 'circle', 'triangle', 'star', 'cross', 'hexagon', 'trapezoid', 'L', 'T', 'X']
            prediction = model.predict(sample)[0]
            probabilities = model.predict_proba(sample)[0]
            confidence = max(probabilities)
            
            # Lưu kết quả dự đoán vào lịch sử
            prediction_entry = {
                'image': image_array,  # Lưu ảnh để hiển thị
                'shape': shape_names[prediction],
                'prediction_id': prediction,
                'confidence': confidence
            }
            st.session_state['prediction_history'].append(prediction_entry)
            
            # Hiển thị kết quả dự đoán
            st.subheader("Kết Quả Dự Đoán")
            st.write(f"Hình dạng dự đoán: **{shape_names[prediction]}** (ID: {prediction})")
            st.write(f"Độ tin cậy: **{confidence:.2%}**")
            
            # Log kết quả dự đoán vào MLflow
            if st.button("Log Kết Quả Dự Đoán"):
                with mlflow.start_run(run_name=f"Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("model_run_id", selected_run_id)
                    mlflow.log_param("predicted_shape", shape_names[prediction])
                    mlflow.log_param("prediction_id", prediction)
                    mlflow.log_metric("confidence", confidence)
                    st.success("Kết quả dự đoán đã được log vào MLflow!")
    
    # Hiển thị lịch sử dự đoán
    st.subheader("Lịch Sử Dự Đoán")
    if st.session_state['prediction_history']:
        for i, entry in enumerate(st.session_state['prediction_history']):
            st.write(f"**Dự đoán {i+1}:**")
            col1, col2 = st.columns([1, 2])
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(entry['image'], cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
            with col2:
                st.write(f"Hình dạng: **{entry['shape']}** (ID: {entry['prediction_id']})")
                st.write(f"Độ tin cậy: **{entry['confidence']:.2%}**")
            st.write("---")
    else:
        st.write("Chưa có dự đoán nào.")