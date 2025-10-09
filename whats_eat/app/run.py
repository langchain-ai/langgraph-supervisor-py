# app/run.py
from whats_eat.app.supervisor_app import build_app

# Export compiled app for LangGraph Server discovery
app = build_app()

if __name__ == "__main__":
    # Example manual run (optional):
    # for chunk in app.stream({
    #     "messages": [{"role": "user", "content": "Find ramen near Tanjong Pagar and show photos"}]}
    # ):
    #     print(chunk)
    pass
