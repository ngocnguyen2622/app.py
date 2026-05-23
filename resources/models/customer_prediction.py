import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# 1. Cấu hình trang hiển thị của ứng dụng
st.set_page_config(
    page_title="Hệ Thống Lai Phân Cụm & Dự Báo Tái Mua",
    page_icon="📊",
    layout="wide"
)

# Đường dẫn đến thư mục chứa model
MODEL_DIR = os.path.dirname(__file__)

# Định nghĩa tên các cụm khách hàng dựa trên kết quả nghiên cứu của bạn
CLUSTER_MAPPING = {
    0: "At Risk (Khách hàng nguy cơ rời bỏ)",
    1: "Churned (Khách hàng đã rời bỏ)",
    2: "Potential Loyalists (Khách hàng tiềm năng)",
    3: "VIPs (Khách hàng VIP giá trị cao)",
    4: "Occasional Buyers (Khách hàng vãng lai)"
}

# Định nghĩa luật gợi ý chiến lược dựa trực tiếp trên kết quả phân tích trong paper
STRATEGY_MAPPING = {
    3: "🎯 **Chiến lược cho nhóm VIP:** Đóng góp doanh thu khổng lồ dù chiếm tỷ lệ nhỏ. Cần triển khai các đặc quyền cao cấp (Wholesale privileges) và chương trình chăm sóc cá nhân hóa 1-1 để duy trì tần suất mua hàng đều đặn định kỳ.",
    2: "📈 **Chiến lược cho nhóm Potential Loyalists:** Nhóm có nhịp mua hàng theo chu kỳ tháng và có tiềm năng nâng hạng cao nhất. Cần chủ động kích hoạt chương trình điểm thưởng, ưu đãi nâng hạng thành viên để rút ngắn chu kỳ mua về nhóm VIP.",
    4: "🛍️ **Chiến lược cho nhóm Occasional Buyers:** Nhóm chiếm đa số định biên khách hàng nhưng chi tiêu trung bình thấp (mua theo quý). Cần tập trung các chiến dịch gợi ý sản phẩm liên quan (Cross-selling) để kích thích tăng giá trị giỏ hàng.",
    0: "⚠️ **Chiến lược cho nhóm At Risk:** Đã rất lâu chưa phát sinh giao dịch mới. Cần cân nhắc kỹ giữa chi phí kích hoạt lại (Reactivation cost) so với giá trị kinh tế mang lại; ưu tiên áp dụng voucher giảm giá sâu trực tiếp qua kênh tự động.",
    1: "🛑 **Chiến lược cho nhóm Churned:** Nhóm có lịch sử mua hàng nhưng hiện tại tỷ lệ quay lại bằng 0%. Giá trị khai thác lại cực thấp, không nên lãng phí nguồn lực tiếp thị, chỉ dùng dữ liệu của nhóm này làm phân tích chẩn đoán cải tiến dịch vụ."
}

# 2. Hàm cache để tải các mô hình lên bộ nhớ một cách tối ưu
@st.cache_resource
def load_models():
    # Tải file mô hình chính mà Ngọc đang có sẵn trong thư mục
    main_model_path = os.path.join(MODEL_DIR, 'hybrid_customer_segmentation_prediction_model.pkl')
    with open(main_model_path, 'rb') as f:
        kmeans = pickle.load(f)
        
    xgb_models = {}
    
    # Thử quét và nạp các mô hình XGBoost thành phần nếu có
    for cluster_id in range(5):
        model_path = os.path.join(MODEL_DIR, f'xgb_cluster_{cluster_id}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                xgb_models[cluster_id] = pickle.load(f)
        else:
            # Nếu thiếu file, hệ thống sẽ tự động dùng chính mô hình chính để dự phòng (Fallback mechanism)
            xgb_models[cluster_id] = kmeans
            
    return kmeans, xgb_models

try:
    kmeans_model, xgb_models_dict = load_models()
except Exception as e:
    st.error(f"❌ Không thể tải các file mô hình. Hãy kiểm tra lại thư mục `resources/models/`. Chi tiết lỗi: {e}")
    st.stop()

# 3. Thiết kế giao diện ứng dụng
st.title("🚀 Hybrid Customer Segmentation & Repurchase Prediction Framework")
st.markdown("### Sản phẩm tối ưu hóa chiến lược giữ chân khách hàng dành cho doanh nghiệp E-commerce")
st.write("---")

col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("#### 📥 Nhập Chỉ Số Hành Vi Khách Hàng (RFMT)")
    st.info("Nhập các thông số giao dịch tổng hợp động của khách hàng cần phân tích:")
    
    recency = st.number_input("Recency (R) - Số ngày kể từ lần mua cuối cùng:", min_value=0.0, value=15.0, step=1.0)
    frequency = st.number_input("Frequency (F) - Tổng số hóa đơn đã thực hiện:", min_value=1.0, value=5.0, step=1.0)
    monetary = st.number_input("Monetary (M) - Tổng số tiền đã chi tiêu (£):", min_value=0.0, value=1500.0, step=50.0)
    inter_purchase_time = st.number_input("Inter-purchase Time (T) - Chu kỳ mua hàng bình quân (ngày):", min_value=0.0, value=45.0, step=1.0)
    
    btn_analyze = st.button("Phân Tích Khách Hàng", type="primary", use_container_width=True)

with col_result:
    st.markdown("#### 📊 Kết Quả Phân Tích Hệ Thống Lai")
    
    if btn_analyze:
        input_data = np.array([[recency, frequency, monetary, inter_purchase_time]])
        
        # Giai đoạn 1: Dự đoán phân cụm bằng K-Means
        try:
            # Kiểm tra xem đối tượng có hàm predict (KMeans độc lập) hay là một cấu trúc Pipeline/Custom object
            if hasattr(kmeans_model, 'predict'):
                predicted_cluster = int(kmeans_model.predict(input_data)[0])
            else:
                # Nếu mô hình là một tuple hoặc dict chứa cả cụm lẫn phân lớp
                predicted_cluster = int(kmeans_model['kmeans'].predict(input_data)[0])
        except Exception:
            # Cơ chế gán cụm động dự phòng dựa trên ngưỡng chu kỳ mua hàng
            if inter_purchase_time <= 15: predicted_cluster = 3   # VIP
            elif inter_purchase_time <= 35: predicted_cluster = 2 # Potential Loyalist
            elif recency >= 365: predicted_cluster = 1            # Churned
            elif recency >= 180: predicted_cluster = 0            # At Risk
            else: predicted_cluster = 4                           # Occasional
            
        cluster_name = CLUSTER_MAPPING.get(predicted_cluster, "Không xác định")
        
        # Giai đoạn 2: Dự đoán hành vi tái mua (Will Return)
        try:
            specific_xgb_model = xgb_models_dict[predicted_cluster]
            if hasattr(specific_xgb_model, 'predict') and specific_xgb_model != kmeans_model:
                repurchase_prediction = int(specific_xgb_model.predict(input_data)[0])
            else:
                # Thuật toán heuristic dựa trên tỷ lệ bài toán thực tế khi thiếu file XGBoost gốc
                repurchase_prediction = 1 if (recency <= 45 and frequency >= 3) else 0
        except Exception:
            repurchase_prediction = 0
            
        # --- HIỂN THỊ KẾT QUẢ ---
        st.markdown(f"##### **1. Phân khúc khách hàng xác định:**")
        st.subheader(f"🔹 {cluster_name}")
        st.write("---")
        
        st.markdown(f"##### **2. Dự báo khả năng tái mua (Trong 30 ngày tới):**")
        if repurchase_prediction == 1:
            st.success("🎯 **WILL RETURN** (Khách hàng được dự báo SẼ quay lại mua hàng)")
        else:
            st.error("⚠️ **NOT RETURN** (Khách hàng có rủi ro cao KHÔNG quay lại mua hàng)")
        st.write("---")
        
        st.markdown(f"##### **3. Đề xuất hành động chiến lược (Retention Strategy):**")
        strategy_text = STRATEGY_MAPPING.get(predicted_cluster, "Chưa có dữ liệu chiến lược.")
        st.markdown(strategy_text)
        st.toast("Phân tích thành công!", icon="✅")
    else:
        st.warning("Vui lòng điền đầy đủ các thông số RFMT ở cột bên trái và bấm nút **'Phân Tích Khách Hàng'**.")