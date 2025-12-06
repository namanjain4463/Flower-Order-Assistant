import streamlit as st
import os
import sys

# Flag to track if all services initialized successfully
setup_successful = False 

# --- Defensive Import Check (Prevents White Screen Crash) ---
try:
    # 1. Check for ALL required secrets before proceeding
    required_secrets = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "OPENAI_MODEL"]
    for key in required_secrets:
        # Check if the key is present in Streamlit secrets
        if key not in st.secrets:
            # Raise an explicit error to halt execution gracefully
            raise KeyError(f"Missing required secret key: '{key}'. Please configure your secrets to run the application.")

    # 2. Proceed with functional imports (which use the secrets)
    # The imports below instantiate LLM and Neo4j connections immediately.
    from utils import write_message
    from agent import generate_response

    # If all imports and instantiations succeed, set the flag
    setup_successful = True

except Exception as e:
    # --- ERROR DISPLAY ---
    # If any error occurs during module loading or secret access, display it visibly
    # We use a basic config here in case the main one causes issues
    st.set_page_config(page_title="App Initialization Error", page_icon="❌")
    st.error("🚨 Application Setup Failed")
    st.markdown("---")
    st.write("The chat interface could not load due to an initialization error. This is often caused by invalid credentials (Neo4j, OpenAI) or missing dependency packages.")
    st.exception(e)
    # We do NOT exit here, allowing Streamlit to finish rendering the error message
    # but the setup_successful flag is False, preventing the rest of the app from running.

# ====================================================================
# MAIN APPLICATION LOGIC (Only runs if setup_successful is True)
# ====================================================================

if setup_successful:
    # Page Config
    st.set_page_config(
        page_title="Flower Order Assistant",
        page_icon="🌸",
        layout="centered"
    )

    # Set up Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm the Flower Ordering Chatbot. We offer flowers in red, orange, "
                    "pink, purple, and white. Just tell me the colors and quantities you'd "
                    "like, and I'll prepare the order for you!"
                )
            }
        ]
    # Holds BOM text if present
    if "pending_bom" not in st.session_state:
        st.session_state.pending_bom = None

    # Holds last-generated CSV path (for download)
    if "last_csv_path" not in st.session_state:
        st.session_state.last_csv_path = None

    # Submit handler
    def handle_submit(message):
        """
        Handles a user message → passes into the LLM agent → stores
        BOM and CSV filenames when returned.
        """
        with st.spinner("Thinking..."):
            # Initial response generation
            response = generate_response(message)
            csv_path = None
            
            # Store BOM temporarily if detected (Step 4 response)
            if "BILL OF MATERIALS" in response.upper():
                st.session_state.pending_bom = response
            
            # Detect confirmation input (Step 5 execution)
            if message.strip().lower() in ["yes", "y", "confirm", "correct"] and st.session_state.pending_bom:
                # Pass confirmation to the agent
                response = generate_response("CONFIRM_ORDER") 
                st.session_state.pending_bom = None
            
            # Detect CSV creation/Robot Execution
            if "CSV file created at:" in response:
                # Cleanly extract path from the agent's text response
                csv_path = response.split("CSV file created at:")[-1].split("---")[0].strip() 
                st.session_state.last_csv_path = csv_path
                
            # Display assistant response
            write_message("assistant", response)

    # Display messages in Session State
    for message in st.session_state.messages:
        write_message(message['role'], message['content'], save=False)

    # Optional: Add a download button for the last generated CSV
    if st.session_state.last_csv_path and os.path.exists(st.session_state.last_csv_path):
        with open(st.session_state.last_csv_path, "r") as f:
            st.download_button(
                label="Download Last Bill of Order (CSV)",
                data=f.read(),
                file_name=os.path.basename(st.session_state.last_csv_path),
                mime="text/csv",
                key='download_csv_button'
            )


    # Handle any user input
    if question := st.chat_input("Enter your flower order…"):
        # Display user message in chat message container
        write_message('user', question)

        # Generate a response
        handle_submit(question)