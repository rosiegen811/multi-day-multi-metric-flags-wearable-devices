# CDC PLACES short sleep, Massachusetts

Saved 2026-10-18 for Hackachusetts.

CDC PLACES 2022 crude prevalence of short sleep, under 7 hours, for Massachusetts counties. This is a population survey, not wearable data.

Rows in data.csv: 14

Columns: year, county, prevalence, low, high, population

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://data.cdc.gov/resource/swc5-untb.json?$select=year,locationname,data_value,low_confidence_limit,high_confidence_limit,totalpopulation&stateabbr=MA&measure=Short%20sleep%20duration%20among%20adults&data_value_type=Crude%20prevalence&$limit=50

Check the source license before publishing a project.
