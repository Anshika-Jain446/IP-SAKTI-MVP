import streamlit as st
import requests


# =========================================================
# CONFIG
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IP-SAKTI",
    page_icon="⚖️",
    layout="centered",
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "session_id": None,
    "question": None,
    "history": [],
    "completed": False,
    "last_error": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPERS
# =========================================================

def extract_question(data: dict):
    """
    The backend may return the next question under different keys.
    Prefer 'question', then common alternatives.
    """
    if not isinstance(data, dict):
        return None

    for key in ("question", "current_question", "next_question", "prompt"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def normalize_question(data: dict, fallback_number: int):
    """
    Normalize the backend response so Streamlit always has:
      - question
      - question_number
      - total_questions
      - completed
      - session_id (when supplied)
    """
    if not isinstance(data, dict):
        data = {}

    question = extract_question(data)

    # Prevent the input widget from getting the same key forever.
    question_number = data.get("question_number")
    if not isinstance(question_number, int) or question_number < 1:
        question_number = fallback_number

    total_questions = data.get("total_questions", 0)
    if not isinstance(total_questions, int) or total_questions < 0:
        total_questions = 0

    return {
        **data,
        "question": question or "",
        "question_number": question_number,
        "total_questions": total_questions,
        "completed": bool(data.get("completed", False)),
    }


def get_next_question(session_id: str, fallback_number: int):
    """
    IMPORTANT FIX:
    /api/conversation/answer may save the answer but not return
    the next question. In that case explicitly call /next.
    """
    response = requests.get(
        f"{BACKEND_URL}/api/conversation/{session_id}/next",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    return normalize_question(data, fallback_number)


def start_assessment():
    """
    Start a fresh conversation.

    If /start does not include a usable question, fetch it from /next.
    """
    response = requests.post(
        f"{BACKEND_URL}/api/conversation/start",
        json={},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    session_id = data.get("session_id")
    if not session_id:
        raise RuntimeError(
            f"Backend did not return session_id. Response: {data}"
        )

    question_data = normalize_question(data, 1)

    if not question_data["question"] and not question_data["completed"]:
        question_data = get_next_question(session_id, 1)

    return session_id, question_data


def submit_answer(session_id: str, answer: str, next_number: int):
    """
    Submit the answer.

    If the answer endpoint returns the next question, use it.
    If it returns an empty/missing question, call /next.
    """
    response = requests.post(
        f"{BACKEND_URL}/api/conversation/answer",
        json={
            "session_id": session_id,
            "answer": answer,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    question_data = normalize_question(data, next_number)

    completed = bool(data.get("completed", False))

    # Backend has finished the conversation.
    if completed:
        question_data["completed"] = True
        return question_data

    # BUG FIX:
    # The old Streamlit code assumed /answer returned "question".
    # When it did not, the UI displayed a blank Q and reused answer_1.
    if not question_data["question"]:
        question_data = get_next_question(session_id, next_number)

    return question_data


def reset_assessment():
    st.session_state.session_id = None
    st.session_state.question = None
    st.session_state.history = []
    st.session_state.completed = False
    st.session_state.last_error = None


# =========================================================
# TITLE
# =========================================================

st.title("⚖️ IP-SAKTI")
st.subheader("IP / Traditional Knowledge Assessment")

st.write(
    "Answer the questions one by one. "
    "Your conversation history will be kept during the session."
)


# =========================================================
# START CONVERSATION
# =========================================================

if st.session_state.session_id is None:

    st.info("Click below to start your assessment.")

    if st.button(
        "🚀 Start Assessment",
        use_container_width=True,
    ):
        try:
            session_id, question_data = start_assessment()

            st.session_state.session_id = session_id
            st.session_state.question = question_data
            st.session_state.history = []
            st.session_state.completed = question_data.get(
                "completed", False
            )
            st.session_state.last_error = None

            st.rerun()

        except requests.exceptions.HTTPError as e:
            st.error("The IP-SAKTI backend returned an HTTP error.")
            st.code(str(e))

            response = getattr(e, "response", None)
            if response is not None:
                try:
                    st.json(response.json())
                except Exception:
                    pass

        except requests.exceptions.RequestException as e:
            st.error("Could not connect to the IP-SAKTI backend.")
            st.code(str(e))

        except Exception as e:
            st.error("Unexpected error while starting the assessment.")
            st.exception(e)


# =========================================================
# CONVERSATION
# =========================================================

if (
    st.session_state.session_id
    and not st.session_state.completed
):

    question_data = st.session_state.question

    if not question_data:
        st.error("No current question was received from the backend.")

    else:
        question_number = question_data.get(
            "question_number",
            len(st.session_state.history) + 1,
        )

        total_questions = question_data.get(
            "total_questions",
            0,
        )

        question = question_data.get(
            "question",
            "",
        )

        # -----------------------------------------------------
        # SAFETY CHECK
        # -----------------------------------------------------

        if not question:
            st.error(
                "The backend did not provide a question."
            )

            if st.button(
                "🔄 Fetch Current Question",
                use_container_width=True,
            ):
                try:
                    st.session_state.question = get_next_question(
                        st.session_state.session_id,
                        len(st.session_state.history) + 1,
                    )
                    st.rerun()

                except Exception as e:
                    st.error("Could not fetch the current question.")
                    st.exception(e)

        else:

            # -------------------------------------------------
            # PROGRESS
            # -------------------------------------------------

            if total_questions:
                progress_value = question_number / total_questions
                progress_value = min(
                    max(progress_value, 0.0),
                    1.0,
                )

                st.progress(progress_value)

                st.caption(
                    f"Question {question_number} of {total_questions}"
                )

            else:
                st.caption(
                    f"Question {question_number}"
                )

            # -------------------------------------------------
            # CONVERSATION HISTORY
            # -------------------------------------------------

            if st.session_state.history:
                st.markdown("### Conversation")

                for item in st.session_state.history:
                    st.markdown(
                        f"**Q:** {item['question']}"
                    )

                    st.markdown(
                        f"**You:** {item['answer']}"
                    )

                    st.divider()

            # -------------------------------------------------
            # CURRENT QUESTION
            # -------------------------------------------------

            st.markdown("### Current Question")
            st.write(question)

            # -------------------------------------------------
            # ANSWER
            # -------------------------------------------------

            # The key is based on the REAL question number.
            # This prevents Streamlit from reusing the previous
            # answer when the backend response has no question_number.
            answer_key = f"answer_{question_number}"

            answer = st.text_input(
                "Your answer",
                key=answer_key,
            )

            if st.button(
                "Next →",
                use_container_width=True,
            ):
                cleaned_answer = answer.strip()

                if not cleaned_answer:
                    st.warning("Please enter an answer.")

                else:
                    try:
                        next_number = (
                            len(st.session_state.history) + 2
                        )

                        next_question_data = submit_answer(
                            st.session_state.session_id,
                            cleaned_answer,
                            next_number,
                        )

                        # Save exactly once.
                        st.session_state.history.append(
                            {
                                "question": question,
                                "answer": cleaned_answer,
                            }
                        )

                        st.session_state.question = (
                            next_question_data
                        )

                        st.session_state.completed = (
                            next_question_data.get(
                                "completed",
                                False,
                            )
                        )

                        st.session_state.last_error = None

                        st.rerun()

                    except requests.exceptions.HTTPError as e:
                        st.error(
                            "The backend could not process your answer."
                        )
                        st.code(str(e))

                        response = getattr(e, "response", None)
                        if response is not None:
                            try:
                                st.json(response.json())
                            except Exception:
                                pass

                    except requests.exceptions.RequestException as e:
                        st.error(
                            "Could not send your answer to the backend."
                        )
                        st.code(str(e))

                    except Exception as e:
                        st.error(
                            "Unexpected error while submitting the answer."
                        )
                        st.exception(e)


# =========================================================
# COMPLETED
# =========================================================

if (
    st.session_state.session_id
    and st.session_state.completed
):

    st.success("✅ Conversation completed!")

    st.markdown("### Your Answers")

    for item in st.session_state.history:
        st.markdown(
            f"**Q:** {item['question']}"
        )

        st.markdown(
            f"**A:** {item['answer']}"
        )

        st.divider()

    st.info(
        "Your conversation has been collected successfully. "
        "The next step is to connect these answers to "
        "the IP-SAKTI analysis endpoint."
    )

    with st.expander("View collected session data"):
        st.json(
            {
                "session_id": st.session_state.session_id,
                "answers": st.session_state.history,
            }
        )

    if st.button(
        "🔄 Start New Assessment",
        use_container_width=True,
    ):
        reset_assessment()
        st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("System Status")

    try:
        response = requests.get(
            f"{BACKEND_URL}/",
            timeout=5,
        )

        if response.ok:
            st.success("Backend: Connected")

            try:
                backend_data = response.json()

                st.caption(
                    f"API Version: "
                    f"{backend_data.get('version', 'Unknown')}"
                )

            except Exception:
                pass

        else:
            st.warning("Backend: Not ready")

    except Exception:
        st.error("Backend: Offline")

    st.divider()

    st.write(
        f"Backend: `{BACKEND_URL}`"
    )

    if st.session_state.session_id:
        st.write("Session active: ✅")
    else:
        st.write("Session active: ❌")
