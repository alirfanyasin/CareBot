import streamlit as st
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set konfigurasi halaman
st.set_page_config(
    page_title="CareBot",
    page_icon="🤖",
)

# Load model
@st.cache_resource
def load_model():
    model_name = "model_final"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Savve history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Header
st.title("CareBot")
st.write("Konsultasikan keluhanmu ke CareBot")

# Input pengguna
user_input = st.text_input("Anda:", placeholder="Tanyakan sesuatu...", key="input")

# Tambahkan tombol untuk mengirim input
if st.button("Kirim"):
    if user_input:
        # Tambahkan input pengguna ke percakapan
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Proses model untuk memberikan respons
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Tambahkan respons bot ke percakapan
        st.session_state["messages"].append({"role": "bot", "content": bot_response})

    else:
        st.warning("Mohon masukkan sesuatu sebelum mengirim!")

# Tampilkan percakapan
for message_data in st.session_state["messages"]:
    if message_data["role"] == "user":
        message(message_data["content"], is_user=True)
    else:
        message(message_data["content"], is_user=False)
