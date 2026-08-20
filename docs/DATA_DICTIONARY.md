# Data Dictionary

The raw dataset contains 4,653 rows and 17 columns. Missing counts describe `jobs_dataset.csv`.

| Field | Type | Missing | Description |
|---|---|---:|---|
| `job_id` | integer | 0 | Source record identifier |
| `job_title` | text | 0 | Advertised role title |
| `company` | text | 1 | Employer name |
| `location` | text | 0 | Advertised location |
| `salary_min` | number | 2,354 | Lower salary bound in local currency |
| `salary_max` | number | 2,358 | Upper salary bound in local currency |
| `description` | text | 0 | Job-description text |
| `country` | category | 0 | Posting country |
| `search_keyword` | category | 0 | Collection search term |
| `experience_required` | number | 4,166 | Parsed experience; units and plausibility require review |
| `degree_required` | category | 0 | Parsed degree requirement |
| `skills` | text/list | 0 | Serialized identified skills |
| `num_skills` | integer | 0 | Identified-skill count |
| `job_level` | category | 0 | Junior, Mid, Senior, or Lead |
| `is_remote` | boolean | 0 | Remote-posting indicator |
| `salary_avg` | number | 2,354 | Average salary target before USD standardization |
| `has_salary` | boolean | 0 | Salary-presence indicator |

`jobs_dataset.csv` is duplicated at the root because the preserved notebook expects that path. `data/raw/jobs_dataset.csv` is the structured raw-data location; both copies had matching SHA-256 hashes when packaged. Derived files under `data/processed/` are outputs, not raw-data replacements.
