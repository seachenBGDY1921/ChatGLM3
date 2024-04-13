import streamlit as st 
import webbrowser
import sqlite3
from user import User
import os

st.set_page_config(page_title="PMGR", page_icon="🚀" )     
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1532024802178-20dbc87a312a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80");
background-size: 100%;
display: flex;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


script_dir = os.path.dirname(__file__)
db_path = os.path.join(script_dir, 'chat_sessions.db')

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("""CREATE TABLE if not exists pwd_mgr (app_name varchar(20) not null,
                        user_name varchar(50) not null,
                        pass_word varchar(50) not null,
                        email_address varchar(100) not null,
                        url varchar(255) not null,
                    primary key(app_name)       
                    );""")


def insert_data(u):
    with conn:
        c.execute("insert into pwd_mgr values (:app, :user, :pass, :email, :url)", {'app': u.app, 'user': u.username, 'pass': u.password, 'email': u.email, 'url': u.url})
        
def get_cred_by_app(app):
    with conn:
        c.execute("select app_name, user_name, pass_word, email_address, url FROM pwd_mgr where app_name = :name;", {'name': app})
        return c.fetchone()
    
def remove_app_cred(app):
    with conn:
        c.execute("DELETE from pwd_mgr WHERE app_name = :name", {'name': app})
        
def update_password(app,new_pass_word):
    with conn:
        c.execute("update pwd_mgr set pass_word = :pass where app_name = :name", {'name': app, 'pass': new_pass_word})


st.title("密码管理 🔐")
st.markdown('#')

c.execute("select count(*) from pwd_mgr")
db_size = c.fetchone()[0] 

c.execute("select app_name from pwd_mgr")
app_names = c.fetchall()
app_names = [i[0] for i in app_names]

radio_option = st.sidebar.radio("菜单", options=["首页", "新增账户", "更新密码", "删除账户"])

if radio_option == "首页":
    st.subheader("查找凭据 🔎")
    st.markdown("#####")
    if db_size > 0:
        option = st.selectbox('选择应用 📱', app_names)  # 从数据库中获取数据填充
        st.markdown("#####")
        cred = get_cred_by_app(option)
        with st.container():
            st.text(f"用户名 👤")
            st.code(f"{cred[1]}", language="python")
            st.text_input('密码 🔑', value=cred[2], type="password", )
            st.markdown("####")
            url = cred[4]
            if st.button('启动 🚀', use_container_width=True):
                webbrowser.open_new_tab(url=url)
        st.markdown('##')
        with st.expander("更多细节:"):
            st.text(f"电子邮件")
            st.code(f"{cred[3]}", language="python")
            st.text(f"网址")
            st.code(f"{cred[4]}", language="python")
    else:
        st.info('数据库为空。', icon="ℹ️")

if radio_option == "新增账户":
    st.subheader("新增凭据 🗝️")
    st.markdown("####")
    app_name = st.text_input('应用 📱', '难')
    user_name = st.text_input('用户名 👤', '大学生只会吃饭')
    pass_word = st.text_input('密码 🔑', '123456', type="password", )
    email = st.text_input('电子邮件 📧', 'tweety@qq.com')
    url = st.text_input('网站 🔗', 'baidu.com')
    st.markdown("####")
    if st.button('保存 ⏬', use_container_width=True):
        try:
            data = User(app_name, user_name, pass_word, email, url)
            insert_data(data)
            st.success(f"{app_name}的凭据已添加到数据库中！", icon="✅")
        except:
            st.warning('出了些问题！请再试一次。', icon="⚠️")
    st.markdown("####")
    st.info(f"数据库中的凭据数量: {db_size}", icon="💾")

if radio_option == "更新密码":
    st.subheader("更新密码 🔄")
    st.markdown('#####')
    if db_size > 0:
        up_app = st.selectbox('选择你想更新的账户 👇', app_names)
        st.markdown('####')
        new_pass_1 = st.text_input('新密码', '新密码123', type="password", )
        new_pass_2 = st.text_input('确认新密码', '新密码123', type="password", )
        if new_pass_1 == new_pass_2:

            if st.button('更新 ⚡️', use_container_width=True):
                try:
                    update_password(up_app, new_pass_1)
                    st.success(f"{up_app}的密码已更新！", icon="✅")
                except:
                    st.info('数据库为空。请转到“新增账户”以添加数据。', icon="ℹ️")
        else:
            st.warning("密码不匹配！请重试。", icon="⚠️")
    else:
        st.info('数据库为空。', icon="ℹ️")

if radio_option == "删除账户":
    st.subheader("删除凭据 🗑️")
    st.markdown("#####")
    if db_size > 0:
        agree = st.checkbox('查看完整数据库')
        if agree:
            c.execute("select app_name, email_address, url from pwd_mgr")
            results = c.fetchall()
            st.table(results)
        st.markdown('#####')
        delt = st.selectbox('选择你想删除的账户 👇', app_names)
        st.markdown('####')
        if st.button('删除 ❌', use_container_width=True):
            try:
                remove_app_cred(delt)
                st.success(f"{delt}的凭据已从数据库中删除！", icon="✅")
            except:
                st.info('数据库为空。请转到“新增账户”以添加数据。', icon="ℹ️")
    else:
        st.info('数据库为空。', icon="ℹ️")