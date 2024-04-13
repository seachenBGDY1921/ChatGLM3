import streamlit as st
from PIL import Image

# 创建一个文件上传器组件，用户可以上传图片
st.write('请选择一个图片文件作为背景:')
uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

# 检查是否有文件被上传
if uploaded_file is not None:
    # 将上传的图片转换为PIL图像对象
    img = Image.open(uploaded_file)

    # 将PIL图像对象转换为HTML的img标签
    img_html = '<img src="data:image;base64,' + img.tobytes().hex() + '" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"/>'
    st.markdown(img_html, unsafe_allow_html=True)

    # 为了确保背景图片不会覆盖内容，我们可以添加一个透明的遮罩层
    st.markdown(
        '<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255, 255, 255, 0.8); z-index: 1;"></div>',
        unsafe_allow_html=True)

    # 继续添加你的页面内容
    st.title('我的Streamlit应用')
    st.write('这里是你的应用内容。')