# CMS hospital readmissions, Massachusetts

Saved 2026-10-18 for Hackachusetts.

CMS Hospital Readmissions Reduction Program measures for Massachusetts facilities. The live call returns 100 rows. data.csv has the full saved slice.

Rows in data.csv: 330

Columns: facility_id, facility_name, measure, discharges, readmissions, excess_ratio, predicted_rate, expected_rate, start_date, end_date

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://data.cms.gov/provider-data/api/1/datastore/query/9n3s-kdb3/0?limit=100&conditions[0][property]=state&conditions[0][value]=MA&conditions[0][operator]==

Check the source license before publishing a project.
