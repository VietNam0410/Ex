import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_data():
    st.header("Xử Lý Dữ Liệu Hình Học")  # Tiêu đề cho dữ liệu hình học
    
    # Tải file dữ liệu
    st.write("Tải bộ dữ liệu hình học (geometric_X.npy và geometric_y.npy)")
    x_file = st.file_uploader("Tải file geometric_X.npy", type=['npy'])
    y_file = st.file_uploader("Tải file geometric_y.npy", type=['npy'])
    
    if x_file is not None and y_file is not None:
        X = np.load(x_file)  # Tải mảng ảnh (10000, 28, 28)
        y = np.load(y_file)  # Tải mảng nhãn (10000,)
        
        # Kiểm tra kích thước dữ liệu
        if X.shape != (10000, 28, 28) or y.shape != (10000,):
            st.error("Dữ liệu không đúng định dạng: X (10000, 28, 28), y (10000,)!")
            return None
        
        # Lưu dữ liệu vào session
        st.session_state['X'] = X
        st.session_state['y'] = y
        
        # Hiển thị thông tin cơ bản
        st.write(f"Số mẫu: {X.shape[0]}")
        st.write(f"Kích thước ảnh: {X.shape[1]}x{X.shape[2]}")
        st.write(f"Số lớp: {len(np.unique(y))}")
        
        # Hiển thị ảnh mẫu
        st.subheader("Minh Họa Dữ Liệu Hình Học")
        sample_idx = st.slider("Chọn mẫu để xem", 0, 9999, 0)  # Chọn mẫu từ 0-9999
        fig, ax = plt.subplots()
        ax.imshow(X[sample_idx], cmap='gray')
        ax.set_title(f"Hình dạng: {y[sample_idx]}")
        ax.axis('off')
        st.pyplot(fig)
        
        return X, y
    return None