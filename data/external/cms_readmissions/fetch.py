"""Save a fresh copy of this Hackachusetts starter source."""
import pathlib
import urllib.parse
import urllib.request

URL = "https://data.cms.gov/provider-data/api/1/datastore/query/9n3s-kdb3/0?limit=100&conditions[0][property]=state&conditions[0][value]=MA&conditions[0][operator]=="
request = urllib.request.Request(URL, headers={"User-Agent": "HackachusettsKit"})
with urllib.request.urlopen(request, timeout=90) as response:
    payload = response.read()
suffix = pathlib.Path(urllib.parse.urlparse(URL).path).suffix
path = pathlib.Path("fresh" + suffix) if suffix in {".csv", ".txt", ".json"} else pathlib.Path("fresh.json")
path.write_bytes(payload)
print(f"wrote {path} ({len(payload)} bytes)")
