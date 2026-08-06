"""PawPal+ Streamlit UI.

Presentation layer over two things: the deterministic logic in pawpal_system.py
and the AI layer in pawpal_ai/. All pet/task state lives in a single Owner object
stored in st.session_state so it survives Streamlit's top-to-bottom reruns.

The AI never edits the schedule directly. It produces a proposal, the proposal is
rendered for review, and a human presses Approve before any Task is created.
"""

import streamlit as st

from pawpal_ai.agent import CarePlanAgent
from pawpal_ai.config import load_settings
from pawpal_system import Owner, Pet, Scheduler, Task

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")
st.caption("A pet-care planner: add pets, schedule tasks, and let the AI draft a grounded plan.")


# --- Application memory -----------------------------------------------------
# Streamlit reruns this script on every interaction. Store the Owner once so
# pets and tasks persist across reruns instead of being rebuilt empty.
if "owner" not in st.session_state:
    st.session_state.owner = Owner(name="Jordan")
if "proposal" not in st.session_state:
    st.session_state.proposal = None
if "answer" not in st.session_state:
    st.session_state.answer = None

owner: Owner = st.session_state.owner


@st.cache_resource
def get_agent() -> CarePlanAgent:
    """Build the agent once per session; it loads and indexes the corpus."""
    return CarePlanAgent()


settings = load_settings()
agent = get_agent()


# --- Sidebar: system status -------------------------------------------------
with st.sidebar:
    st.subheader("System status")
    if settings.is_live:
        st.success(f"AI: {settings.describe()}")
    else:
        st.warning("AI: offline stub")
        st.caption(
            "No API key found, so plans come from fixed rules and ignore your "
            "constraints. Copy `.env.example` to `.env` and add a free key from "
            "[aistudio.google.com](https://aistudio.google.com)."
        )
    st.caption(f"Knowledge base: {len(agent.retriever.chunks)} passages indexed")
    st.divider()
    st.caption(
        "The AI proposes; you approve. Nothing is added to your schedule without "
        "your confirmation."
    )


# --- Owner ------------------------------------------------------------------
owner.name = st.text_input("Owner name", value=owner.name)

st.divider()

# --- Add a pet --------------------------------------------------------------
st.subheader("🐶 Add a pet")
with st.form("add_pet", clear_on_submit=True):
    pet_name = st.text_input("Pet name", value="")
    species = st.selectbox("Species", ["dog", "cat", "other"])
    if st.form_submit_button("Add pet"):
        if pet_name.strip():
            owner.add_pet(Pet(name=pet_name.strip(), species=species))
            st.success(f"Added {pet_name.strip()} ({species}).")
        else:
            st.warning("Please enter a pet name.")

if owner.pets:
    st.caption("Current pets: " + ", ".join(f"{p.name} ({p.species})" for p in owner.pets))
else:
    st.info("No pets yet. Add one above.")

st.divider()

# --- AI planner -------------------------------------------------------------
st.subheader("🤖 Plan my day with AI")

if not owner.pets:
    st.info("Add a pet first, then the AI can plan for it.")
else:
    request = st.text_area(
        "Describe what you need",
        placeholder=(
            "e.g. Biscuit is a 9-month-old puppy and I work 9 to 5. "
            "Plan a realistic day that doesn't leave him alone too long."
        ),
        height=80,
    )

    if st.button("Generate plan", type="primary"):
        if not request.strip():
            st.warning("Describe what you'd like planned.")
        else:
            with st.spinner("Retrieving guidance and drafting a plan..."):
                st.session_state.proposal = agent.plan(owner, request)

    proposal = st.session_state.proposal
    if proposal is not None:
        if proposal.error and not proposal.tasks:
            st.error(proposal.error)
        else:
            if proposal.degraded:
                st.warning(
                    "This plan came from the offline fallback, not a language model. "
                    "It ignores your stated constraints."
                )

            confidence_label = (
                "high" if proposal.confidence >= 0.7
                else "medium" if proposal.confidence >= 0.4
                else "low"
            )
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Confidence", f"{proposal.confidence:.2f}", confidence_label)
            col_b.metric("Tasks proposed", len(proposal.tasks))
            col_c.metric("Self-repairs", proposal.repairs)

            if proposal.summary:
                st.write(proposal.summary)

            st.table(
                [
                    {
                        "Time": task.time,
                        "Task": task.description,
                        "Pet": task.pet,
                        "Frequency": task.frequency,
                        "Why": task.reason or "—",
                    }
                    for task in proposal.tasks
                ]
            )

            if proposal.sources:
                st.caption("Grounded in: " + ", ".join(f"`{s}`" for s in proposal.sources))
            else:
                st.caption("⚠️ No knowledge-base passage supported this plan.")

            for caveat in proposal.caveats:
                st.info(f"Note: {caveat}")

            with st.expander("Show the agent's reasoning trace"):
                st.code("\n".join(proposal.trace.as_lines()), language="text")

            # --- the human-in-the-loop checkpoint ---------------------------
            approve, discard = st.columns(2)
            if approve.button("✅ Approve and add to schedule", disabled=proposal.committed):
                created = agent.commit(owner, proposal)
                st.success(f"Added {len(created)} task(s) to the schedule.")
                st.session_state.proposal = None
                st.rerun()
            if discard.button("🗑️ Discard"):
                st.session_state.proposal = None
                st.rerun()

st.divider()

# --- AI question answering --------------------------------------------------
st.subheader("💬 Ask a pet-care question")
question = st.text_input(
    "Question", placeholder="e.g. How often should I brush a long-haired cat?"
)
if st.button("Ask"):
    if question.strip():
        with st.spinner("Searching the knowledge base..."):
            st.session_state.answer = agent.answer(question, owner)
    else:
        st.warning("Type a question first.")

answer = st.session_state.answer
if answer is not None:
    if answer.error:
        st.error(answer.error)
    else:
        st.write(answer.answer)
        if answer.sources:
            st.caption(
                f"Sources: {', '.join(f'`{s}`' for s in answer.sources)} · "
                f"confidence {answer.confidence:.2f}"
            )

st.divider()

# --- Add a task manually ----------------------------------------------------
st.subheader("📝 Schedule a task manually")
if not owner.pets:
    st.info("Add a pet first, then you can schedule tasks for it.")
else:
    with st.form("add_task", clear_on_submit=True):
        pet_choice = st.selectbox("Pet", [p.name for p in owner.pets])
        description = st.text_input("Task description", value="Morning walk")
        time = st.text_input("Time (HH:MM)", value="08:00")
        frequency = st.selectbox("Frequency", ["once", "daily", "weekly"])
        if st.form_submit_button("Add task"):
            pet = next(p for p in owner.pets if p.name == pet_choice)
            pet.add_task(Task(description=description, time=time, frequency=frequency))
            st.success(f"Scheduled '{description}' at {time} for {pet_choice}.")

st.divider()

# --- Today's schedule -------------------------------------------------------
st.subheader("🗓️ Today's schedule")
scheduler = Scheduler(owner)

# Conflict warnings surface at the top so the owner sees double-bookings first.
for warning in scheduler.detect_conflicts():
    st.warning(f"⚠️ {warning}")

# Filters use the Scheduler's methods; default view is time-sorted.
col_a, col_b = st.columns(2)
with col_a:
    pet_filter = st.selectbox("Filter by pet", ["All pets"] + [p.name for p in owner.pets])
with col_b:
    status_filter = st.selectbox("Filter by status", ["All", "Outstanding", "Done"])

pairs = scheduler.sort_by_time()
if pet_filter != "All pets":
    pairs = [pr for pr in pairs if pr[0].name == pet_filter]
if status_filter == "Outstanding":
    pairs = [pr for pr in pairs if not pr[1].completed]
elif status_filter == "Done":
    pairs = [pr for pr in pairs if pr[1].completed]

if pairs:
    st.table(
        [
            {
                "Time": task.time,
                "Task": task.description,
                "Pet": pet.name,
                "Frequency": task.frequency,
                "Status": "✅ done" if task.completed else "⏳ todo",
            }
            for pet, task in pairs
        ]
    )
else:
    st.info("No tasks match the current filters.")
