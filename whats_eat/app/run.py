# app/run.py
from whats_eat.app.supervisor_app import build_app

if __name__ == "__main__":
    app = build_app()
    for chunk in app.stream({"messages": [{"role": "user", "content": "Find ramen near Tanjong Pagar and show photos"}]}):
        print(chunk)
