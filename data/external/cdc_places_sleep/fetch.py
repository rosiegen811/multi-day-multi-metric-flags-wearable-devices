"""Save a fresh copy of this Hackachusetts starter source."""
import pathlib
import urllib.parse
import urllib.request

URL = "https://data.cdc.gov/resource/swc5-untb.json?$select=year,locationname,data_value,low_confidence_limit,high_confidence_limit,totalpopulation&stateabbr=MA&measure=Short%20sleep%20duration%20among%20adults&data_value_type=Crude%20prevalence&$limit=50"
request = urllib.request.Request(URL, headers={"User-Agent": "HackachusettsKit"})
with urllib.request.urlopen(request, timeout=90) as response:
    payload = response.read()
suffix = pathlib.Path(urllib.parse.urlparse(URL).path).suffix
path = pathlib.Path("fresh" + suffix) if suffix in {".csv", ".txt", ".json"} else pathlib.Path("fresh.json")
path.write_bytes(payload)
print(f"wrote {path} ({len(payload)} bytes)")
