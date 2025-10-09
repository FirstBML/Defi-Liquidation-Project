import requests
import time
import os

# Wait a few seconds for ngrok to start
time.sleep(2)

try:
    # Fetch ngrok tunnels
    res = requests.get("http://127.0.0.1:4040/api/tunnels")
    tunnels = res.json()["tunnels"]

    # Find the HTTPS tunnel
    public_url = None
    for t in tunnels:
        if t["public_url"].startswith("https"):
            public_url = t["public_url"]
            break

    if not public_url:
        raise Exception("No active HTTPS tunnel found")

    # Define the path to your frontend .env file
    env_path = "../your-frontend-project/.env.local"  # adjust this path!

    # Update .env.local file
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    with open(env_path, "w") as f:
        found = False
        for line in lines:
            if line.startswith("NEXT_PUBLIC_API_BASE="):
                f.write(f"NEXT_PUBLIC_API_BASE={public_url}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"NEXT_PUBLIC_API_BASE={public_url}\n")

    print(f"✅ Updated .env.local with {public_url}")

except Exception as e:
    print(f"❌ Error: {e}")
