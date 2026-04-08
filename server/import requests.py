import requests
url = "https://threejs.org/docs/"
try:
    r = requests.get(url, timeout=20, headers={"User-Agent":"curl/7.78.0"})
    print("status:", r.status_code, "bytes:", len(r.text))
    print(r.text[:500])
except Exception as e:
    print("ERROR:", e)