import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from llm import llm
import json

def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)

def get_session_id():
    return get_script_run_ctx().session_id


def extract_quantities_llm(user_text, matched_colors):
    """
    Ask the LLM to infer quantities for each color.
    """

    prompt = f"""
    Extract quantities for the following flower colors from this text:
    COLORS: {matched_colors}
    TEXT: "{user_text}"

    Return JSON ONLY in this form:
    [
        {{"color": "red", "quantity": 5}},
        {{"color": "white", "quantity": 2}}
    ]
    """

    response = llm.invoke(prompt)
    return json.loads(response)