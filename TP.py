#!/usr/bin/env python
# coding: utf-8

# In[52]:


import requests
import pandas as pd

# Settings
year = 2023
headers = {"User-Agent": "Mozilla/5.0"}

# 1) Get all D-I teams and their IDs via ESPN teams API
teams_url = (
    "https://site.web.api.espn.com/apis/site/v2/"
    "sports/basketball/mens-college-basketball/teams?region=us&lang=en&limit=1000"
)
resp = requests.get(teams_url, headers=headers)
resp.raise_for_status()
teams_json = resp.json()['sports'][0]['leagues'][0]['teams']
team_list = [(t['team']['id'], t['team']['displayName']) for t in teams_json]
print(f"🏀 Found {len(team_list)} D-I teams via teams API.")

# 2) Function to fetch per-game stats + WinPct for one team
def fetch_team_stats(team_id, team_name, year):
    # Scrape per-game HTML tables
    stats_url = f"https://www.espn.com/mens-college-basketball/team/stats/_/id/{team_id}/year/{year}"
    r = requests.get(stats_url, headers=headers)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    names_df, stats_df = tables[0], tables[1]
    stats_df.columns = [c if not isinstance(c, tuple) else c[1] for c in stats_df.columns]
    stats_df['Player'] = names_df['Name']
    stats_df['Team']   = team_name
    # Fetch WinPct from summary JSON
    sum_url = (
        "https://site.web.api.espn.com/apis/site/v2/"
        "sports/basketball/mens-college-basketball/summary"
        f"?team={team_id}&season={year}&seasontype=2"
    )
    j = requests.get(sum_url, headers=headers).json()
    win_pct = 0
    for rec in j.get('team', {}).get('record', {}).get('items', []):
        for s in rec.get('stats', []):
            if s.get('name') in ('winPct','WinPct'):
                win_pct = float(s.get('value'))
                break
        if win_pct:
            break
    stats_df['WinPct'] = win_pct
    return stats_df

# 3) Loop through all teams, collect stats, and track skips
all_stats = []
skipped = []
for tid, name in team_list:
    try:
        df_team = fetch_team_stats(tid, name, year)
        all_stats.append(df_team)
    except Exception as e:
        skipped.append(name)
print(f"Finished fetching; skipped {len(skipped)} teams.")
if skipped:
    print("Skipped teams:", skipped)

# 4) Combine into one DataFrame and check coverage
if not all_stats:
    raise RuntimeError("No team stats fetched – check API or year selection.")
raw = pd.concat(all_stats, ignore_index=True)
print(f"🏀 Loaded {len(raw)} player-rows for {year}")
expected = len(team_list)
scraped = raw['Team'].nunique()
print(f"Teams expected: {expected}, Teams scraped: {scraped}")
missing = set([n for _,n in team_list]) - set(raw['Team'].unique())
if missing:
    print(f"Missing teams (no data): {missing}")

# 5) Define stats for TalentScore
desired = ['GP','MIN','PTS','AST','REB','STL','BLK','TO','FG%','3P%','FT%','WinPct']
stats = [s for s in desired if s in raw.columns]
print("Using stats:", stats)

# 6) Clean & filter usage
df = raw.copy()
for s in stats:
    df[s] = pd.to_numeric(df[s], errors='coerce')
# Fill missing percentages and winPct
for p in ['FG%','3P%','FT%','WinPct']:
    if p in df.columns:
        df[p] = df[p].fillna(0)
# Usage filters
df = df[(df['MIN'] >= 15) & (df['GP'] >= 5)]

# 7) Advanced metrics: totals & per-40
for s in ['PTS','REB','AST','STL','BLK','TO']:
    df[f'Total_{s}'] = df[s] * df['GP']
    df[f'{s}_per40'] = df[s] / df['MIN'] * 40

# 8) Compute z-scores & invert turnovers
z = (df[stats] - df[stats].mean()) / df[stats].std()
if 'TO' in stats:
    z['TO'] = -z['TO']
for s in stats:
    df['z_' + s] = z[s]

# 9) TalentScore & Top20
df['TalentScore'] = z.sum(axis=1)
top20 = df.sort_values('TalentScore', ascending=False).head(20)
print(top20[['Player','Team','TalentScore']])


# In[53]:


import pandas as pd

# 1) Allow pandas to print every row & column in the notebook
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 2) Display the entire DataFrame
display(df)

# 3) Export to CSV for external review
df.to_csv('all_players.csv', index=False)
print("Wrote full table to all_players.csv")


# In[ ]:





# In[56]:


import requests
import pandas as pd

# Settings
year = 2023
headers = {"User-Agent": "Mozilla/5.0"}

# 1) Get all D-I teams via ESPN teams API
teams_url = (
    "https://site.web.api.espn.com/apis/site/v2/"
    "sports/basketball/mens-college-basketball/teams?region=us&lang=en&limit=1000"
)
resp = requests.get(teams_url, headers=headers)
resp.raise_for_status()
teams_json = resp.json()['sports'][0]['leagues'][0]['teams']
team_list = [(t['team']['id'], t['team']['displayName']) for t in teams_json]
print(f"🏀 Found {len(team_list)} D-I teams via teams API.")

# 2) Fetch per-game stats + WinPct for one team
def fetch_team_stats(team_id, team_name, year):
    stats_url = f"https://www.espn.com/mens-college-basketball/team/stats/_/id/{team_id}/year/{year}"
    r = requests.get(stats_url, headers=headers)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    names_df, stats_df = tables[0], tables[1]
    stats_df.columns = [c if not isinstance(c, tuple) else c[1] for c in stats_df.columns]
    stats_df['Player'] = names_df['Name']
    stats_df['Team']   = team_name
    # Get WinPct
    sum_url = (
        "https://site.web.api.espn.com/apis/site/v2/"
        "sports/basketball/mens-college-basketball/summary"
        f"?team={team_id}&season={year}&seasontype=2"
    )
    j = requests.get(sum_url, headers=headers).json()
    win_pct = 0
    for rec in j.get('team', {}).get('record', {}).get('items', []):
        for s in rec.get('stats', []):
            if s.get('name') in ('winPct','WinPct'):
                win_pct = float(s.get('value'))
                break
        if win_pct:
            break
    stats_df['WinPct'] = win_pct
    return stats_df

# 3) Loop through teams and collect
all_stats = []
skipped = []
for tid, name in team_list:
    try:
        df_team = fetch_team_stats(tid, name, year)
        all_stats.append(df_team)
    except Exception:
        skipped.append(name)
print(f"Finished fetching; skipped {len(skipped)} teams.")
if skipped:
    print("Skipped teams:", skipped)

# 4) Combine and verify coverage
raw = pd.concat(all_stats, ignore_index=True)
print(f"🏀 Loaded {len(raw)} player-rows for {year}")
print(f"Teams scraped: {raw['Team'].nunique()} of {len(team_list)}")
missing = set(n for _,n in team_list) - set(raw['Team'].unique())
if missing:
    print("Missing teams:", missing)

# 5) Define stats & clean/filter
desired = ['GP','MIN','PTS','AST','REB','STL','BLK','TO','FG%','3P%','FT%','WinPct']
stats = [s for s in desired if s in raw.columns]
print("Using stats:", stats)
df = raw.copy()
for s in stats:
    df[s] = pd.to_numeric(df[s], errors='coerce')
df[['FG%','3P%','FT%','WinPct']] = df[['FG%','3P%','FT%','WinPct']].fillna(0)
df = df[(df['MIN'] >= 15) & (df['GP'] >= 5)]

# 6) Advanced metrics: total & per-40
for s in ['PTS','REB','AST','STL','BLK','TO']:
    df[f'Total_{s}'] = df[s] * df['GP']
    df[f'{s}_per40'] = df[s] / df['MIN'] * 40

# 7) Compute z-scores, invert turnovers
z = (df[stats] - df[stats].mean()) / df[stats].std()
if 'TO' in stats:
    z['TO'] = -z['TO']
for s in stats:
    df['z_' + s] = z[s]

# 8) Calculate TalentScore
df['TalentScore'] = z.sum(axis=1)

# 9) Anchor-based NIL Worth calculation
# Fuzzy-match anchor names in the Player column
toppin_matches = df[df['Player'].str.contains('Toppin', case=False, na=False)]['Player'].unique()
agbim_matches = df[df['Player'].str.contains('Agbim', case=False, na=False)]['Player'].unique()
print('Toppin matches found:', toppin_matches)
print('Agbim matches found:', agbim_matches)
# Ensure exactly one match for each
if len(toppin_matches) != 1 or len(agbim_matches) != 1:
    raise RuntimeError('Could not uniquely identify JT Toppin or Obi Agbim. ' 
                       'Please adjust your substring filters.')
# Use the found exact names
jtt_name = toppin_matches[0]
obi_name = agbim_matches[0]
anchors = {
    jtt_name: 4000000,
    obi_name: 1000000
}
# Lookup their TalentScores
scores = {name: df.loc[df['Player'] == name, 'TalentScore'].iloc[0] for name in anchors}
# Unpack anchor points
a = (anchors[jtt_name] - anchors[obi_name]) / (scores[jtt_name] - scores[obi_name])
b = anchors[jtt_name] - a * scores[jtt_name]
# Compute NIL_Worth
df['NIL_Worth'] = (df['TalentScore'] * a + b).clip(lower=0).round().astype(int)

# 10) Show top 20 with NIL Worth


# In[57]:


# 10) Show top 20 players with TalentScore and anchored NIL_Worth
top20 = df.sort_values('TalentScore', ascending=False).head(20)
print(top20[['Player', 'Team', 'TalentScore', 'NIL_Worth']])


# In[59]:


# Cap NIL worth so JT Toppin remains the maximum
max_worth = anchors[jtt_name]
df['NIL_Worth'] = (df['TalentScore'] * a + b).clip(lower=0, upper=max_worth).round().astype(int)


# In[60]:


# 10) Show top 20 players with TalentScore and anchored NIL_Worth
top20 = df.sort_values('TalentScore', ascending=False).head(20)
print(top20[['Player', 'Team', 'TalentScore', 'NIL_Worth']])


# In[61]:


# 11) Show ALL players sorted by NIL_Worth
df_sorted = df.sort_values('NIL_Worth', ascending=False)
# Display in notebook
display(df_sorted[['Player','Team','TalentScore','NIL_Worth']])
# Export to CSV
output_file = 'players_by_NIL_Worth.csv'
df_sorted.to_csv(output_file, index=False)
print(f"Exported full sorted list to {output_file}")


# In[70]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import display

# — Settings —
year = 2023
headers = {"User-Agent": "Mozilla/5.0"}

# — 1) Fetch list of all D-I teams —
teams_url = (
    "https://site.web.api.espn.com/apis/site/v2/"
    "sports/basketball/mens-college-basketball/teams?region=us&lang=en&limit=1000"
)
resp = requests.get(teams_url, headers=headers); resp.raise_for_status()
teams_json = resp.json()['sports'][0]['leagues'][0]['teams']
team_list  = [(t['team']['id'], t['team']['displayName']) for t in teams_json]

print(f"🏀 Found {len(team_list)} D-I teams")


def fetch_team_stats(team_id, team_name, year):
    """Fetch per-game stats + real Class from the roster page + team WinPct."""
    # a) per-game stats table
    stats_url = f"https://www.espn.com/mens-college-basketball/team/stats/_/id/{team_id}/year/{year}"
    r = requests.get(stats_url, headers=headers); r.raise_for_status()
    tabs = pd.read_html(r.text)
    names_df, stats_df = tabs[0], tabs[1]
    # flatten any multi-index
    names_df.columns  = [c if not isinstance(c, tuple) else c[1] for c in names_df.columns]
    stats_df.columns  = [c if not isinstance(c, tuple) else c[1] for c in stats_df.columns]
    stats_df['Player'] = names_df['Name']
    stats_df['Team']   = team_name

    # b) roster page (to get Class)
    roster_url = f"https://www.espn.com/mens-college-basketball/team/roster/_/id/{team_id}/year/{year}"
    rt = requests.get(roster_url, headers=headers); rt.raise_for_status()
    soup = BeautifulSoup(rt.text, 'html.parser')
    # find the <table> whose header row contains “Class”
    roster_table = None
    for tbl in soup.find_all('table'):
        hdrs = [th.get_text(strip=True).lower() for th in tbl.select('thead th')]
        if 'class' in hdrs:
            roster_table = tbl
            break
    if roster_table is None:
        raise RuntimeError(f"Roster table for {team_name} missing Class column")
    # build dataframe
    cols = [th.get_text(strip=True) for th in roster_table.select('thead th')]
    rows = []
    for tr in roster_table.select('tbody tr'):
        rows.append([td.get_text(strip=True) for td in tr.select('td')])
    roster_df = pd.DataFrame(rows, columns=cols)
    # map Name → Class
    stats_df['Class'] = stats_df['Player'].map(
        dict(zip(roster_df['Name'], roster_df['Class']))
    ).fillna('Unknown')

    # c) team WinPct
    summary_url = (
        "https://site.web.api.espn.com/apis/site/v2/"
        "sports/basketball/mens-college-basketball/summary"
        f"?team={team_id}&season={year}&seasontype=2"
    )
    j = requests.get(summary_url, headers=headers).json()
    win_pct = 0.0
    for rec in j.get('team',{}).get('record',{}).get('items',[]):
        for s in rec.get('stats',[]):
            if s.get('name','').lower() == 'winpct':
                win_pct = float(s.get('value',0))
                break
        if win_pct: break
    stats_df['WinPct'] = win_pct

    return stats_df

# — 2) Loop all teams —
all_stats, skipped = [], []
for tid, name in team_list:
    try:
        all_stats.append(fetch_team_stats(tid, name, year))
    except Exception as e:
        skipped.append((name, str(e)))

print(f"Finished; skipped {len(skipped)} teams")
if skipped:
    for nm, err in skipped:
        print(f" – {nm}: {err}")

# — 3) Combine & clean —
raw = pd.concat(all_stats, ignore_index=True)
print(f"Loaded {len(raw)} player-rows across {raw['Team'].nunique()} teams")

# define stats, filter by usage
desired = ['GP','MIN','PTS','AST','REB','STL','BLK','TO','FG%','3P%','FT%','WinPct']
stats   = [s for s in desired if s in raw.columns]
df = raw.copy()
for s in stats:
    df[s] = pd.to_numeric(df[s], errors='coerce')
df[['FG%','3P%','FT%','WinPct']] = df[['FG%','3P%','FT%','WinPct']].fillna(0)
df = df[(df['MIN']>=15)&(df['GP']>=5)]

# advanced totals & per-40
for s in ['PTS','REB','AST','STL','BLK','TO']:
    df[f'Total_{s}'] = df[s]*df['GP']
    df[f'{s}_per40']  = df[s]/df['MIN']*40

# z-scores & invert turnovers
z = (df[stats] - df[stats].mean())/df[stats].std()
if 'TO' in z: z['TO'] = -z['TO']
for s in stats:
    df['z_'+s] = z[s]

# TalentScore
df['TalentScore'] = z.sum(axis=1)

# anchor NIL worth
jtt = df[df['Player'].str.contains('Toppin',case=False)]['Player'].iloc[0]
obi = df[df['Player'].str.contains('Agbim',case=False)]['Player'].iloc[0]
anchors = {jtt:4_000_000, obi:1_000_000}
s1,s2 = df.loc[df['Player']==jtt,'TalentScore'].iloc[0], df.loc[df['Player']==obi,'TalentScore'].iloc[0]
v1,v2 = anchors[jtt], anchors[obi]
a = (v1-v2)/(s1-s2); b = v1 - a*s1
df['NIL_Worth'] = (df['TalentScore']*a + b).clip(0,v1).round().astype(int)

# age factor
def age_factor(c):
    cl = str(c).lower()
    return 1.2 if 'fr' in cl else 1.15 if 'so' in cl else 1.1 if 'jr' in cl else 1.05 if 'sr' in cl else 1.0
df['AgeFactor'] = df['Class'].apply(age_factor)
df['Weighted_NIL_Worth'] = (df['NIL_Worth']*df['AgeFactor']).round().astype(int)

# Show top 20
top20 = df.sort_values('Weighted_NIL_Worth', ascending=False).head(20)
print(top20[['Player','Class','Team','TalentScore','AgeFactor','Weighted_NIL_Worth']])

# Display/export full list
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
display(df.sort_values('Weighted_NIL_Worth',ascending=False)[['Player','Class','Team','TalentScore','AgeFactor','Weighted_NIL_Worth']])
df.sort_values('Weighted_NIL_Worth',ascending=False)[['Player','Class','Team','TalentScore','AgeFactor','Weighted_NIL_Worth']].to_csv('players_by_weighted_NIL.csv',index=False)


# In[84]:


# Build df_rosters dataframe from roster_list
if roster_list:
    df_rosters = pd.concat(roster_list, ignore_index=True)
else:
    df_rosters = pd.DataFrame(columns=['Player','Class','Team'])


# In[92]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import display

# — 3) Fetch ESPN roster pages via pandas.read_html for Class —
roster_list = []
skipped = []
for team_id, team_name in team_list:
    url = f"https://www.espn.com/mens-college-basketball/team/roster/_/id/{team_id}"
    try:
        tables = pd.read_html(url)
        rf = None
        # Identify the correct table by presence of Name and Yr/Year columns
        for t in tables:
            cols = t.columns
            # Flatten multi-index columns
            flat_cols = []
            for c in cols:
                if isinstance(c, tuple):
                    flat_cols.append(c[1] or c[0])
                else:
                    flat_cols.append(c)
            t.columns = flat_cols
            if 'Name' in t.columns and ('Yr' in t.columns or 'Year' in t.columns):
                rf = t
                break
        if rf is None:
            raise RuntimeError(f"Roster table with Name+Yr/Year not found for {team_name}")
        # Rename to Player and Class
        rf = rf.rename(columns={'Name':'Player', 'Yr':'Class', 'Year':'Class'})
        rf = rf[['Player','Class']]
        rf['Team'] = team_name
        roster_list.append(rf)
    except Exception as e:
        skipped.append((team_name, str(e)))
print(f"Finished ESPN rosters; skipped {len(skipped)} teams.")
for nm, err in skipped[:5]:
    print(f" – {nm}: {err}")

# — 4) Clean & Filter Usage — —
desired = ['GP','MIN','PTS','AST','REB','STL','BLK','TO','FG%','3P%','FT%','WinPct']
stats = [s for s in desired if s in df.columns]
for s in stats:
    df[s] = pd.to_numeric(df[s], errors='coerce')
# usage threshold
df = df[(df['MIN']>=15) & (df['GP']>=5)].copy()

# — 5) Advanced Metrics & z-scores —
for s in ['PTS','REB','AST','STL','BLK','TO']:
    df[f'Total_{s}'] = df[s] * df['GP']
    df[f'{s}_per40'] = df[s] / df['MIN'] * 40
z = (df[stats] - df[stats].mean()) / df[stats].std()
if 'TO' in z: z['TO'] = -z['TO']
for s in stats:
    df['z_'+s] = z[s]

# — 6) TalentScore & NIL Worth —
df['TalentScore'] = z.sum(axis=1)
jtt = df[df['Player'].str.contains('Toppin',case=False)]['Player'].iloc[0]
obi = df[df['Player'].str.contains('Agbim',case=False)]['Player'].iloc[0]
anchors={jtt:4_000_000,obi:1_000_000}
s1,s2=df.loc[df['Player']==jtt,'TalentScore'].iloc[0],df.loc[df['Player']==obi,'TalentScore'].iloc[0]
v1,v2=anchors[jtt],anchors[obi]
a=(v1-v2)/(s1-s2);b=v1-a*s1
df['NIL_Worth'] = (df['TalentScore']*a + b).clip(0,v1).round().astype(int)

# — 7) Age Factor & Weighted NIL —
def age_factor(c):
    cl = str(c).lower()
    if 'fr' in cl: return 1.2
    if 'so' in cl: return 1.15
    if 'jr' in cl: return 1.1
    if 'sr' in cl: return 1.05
    return 1.0
df['AgeFactor'] = df['Class'].apply(age_factor)
df['Weighted_NIL_Worth'] = (df['NIL_Worth'] * df['AgeFactor']).round().astype(int)

# — 8) Show & Export —
top20 = df.sort_values('Weighted_NIL_Worth', ascending=False).head(20)
print(top20[['Player','Class','Team','TalentScore','AgeFactor','Weighted_NIL_Worth']])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
display(df.sort_values('Weighted_NIL_Worth', ascending=False)[['Player','Class','Team','TalentScore','AgeFactor','Weighted_NIL_Worth']])
df.sort_values('Weighted_NIL_Worth', ascending=False)[['Player','Class','Team','TalentScore','AgeFactor','Weighted_NIL_Worth']].to_csv('players_by_weighted_NIL.csv', index=False)
print("Exported to players_by_weighted_NIL.csv")


# In[93]:


pip install sportsipy


# In[99]:


from sportsipy.ncaab.teams import Teams
import pandas as pd


# In[100]:


# Merge on Player + Team
df = pd.merge(df_stats, df_roster, on=['Player','Team'], how='left')

# How many still missing?
miss = df['Class'].isna().sum()
print(f"Players missing Class after merge: {miss}")

# (There should be very few, if any — mostly edge cases like transfers mid-season)
df['Class'] = df['Class'].fillna('Unknown')


# In[110]:


pip install sportsipy


# In[ ]:


import requests
import pandas as pd

# — 0) Pre‑reqs —
# Make sure you already have your `team_list = [(team_id, team_name), …]`
# and your `df_stats` ready.

headers = {"User-Agent": "Mozilla/5.0"}

def fetch_espn_roster_json(team_id, team_name):
    """
    Use ESPN’s common/v3 API to fetch the full roster JSON,
    then extract each player’s name and class year.
    """
    url = (
        "https://site.web.api.espn.com/apis/common/v3/sports/"
        "basketball/mens-college-basketball/teams/"
        f"{team_id}/roster?region=us&lang=en"
    )
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    j = r.json()

    out = []
    for athlete in j.get("athletes", []):
        name = athlete.get("displayName")
        # ESPN’s JSON gives group (e.g. "freshman", "senior", etc.)
        cls  = athlete.get("group", "").capitalize() or "Unknown"
        out.append({"Player": name, "Team": team_name, "Class": cls})
    return pd.DataFrame(out)


# — 1) Build the full roster DF —
roster_dfs = []
skipped = []
for tid, tname in team_list:
    try:
        df_r = fetch_espn_roster_json(tid, tname)
        roster_dfs.append(df_r)
    except Exception as e:
        skipped.append((tname, str(e)))

print(f"✅ Fetched rosters for {len(roster_dfs)} teams, skipped {len(skipped)} teams.")
if skipped:
    for n, err in skipped[:5]:
        print(" –", n, err)

df_roster = pd.concat(roster_dfs, ignore_index=True)
print("Roster columns:", df_roster.columns.tolist())
print("Unique teams:", df_roster['Team'].nunique())


# — 2) Merge into your stats DF —
df = pd.merge(df_stats, df_roster, on=['Player','Team'], how='left')
miss = df['Class'].isna().sum()
print(f"Players missing Class after merge: {miss}")
df['Class'] = df['Class'].fillna('Unknown')


# — 3) Age‐factor mapping & weighted NIL —
def age_factor(c):
    c = str(c).lower()
    if 'fresh' in c:    return 1.2
    if 'soph' in c:     return 1.15
    if 'junior' in c:   return 1.1
    if 'senior' in c:   return 1.05
    return 1.0

df['AgeFactor'] = df['Class'].apply(age_factor)

# Make sure NIL_Worth exists; if not, recompute it like before:
# df['NIL_Worth'] = ...

df['Weighted_NIL_Worth'] = (df['NIL_Worth'] * df['AgeFactor']).round().astype(int)


# — 4) Inspect —
top20 = df.sort_values('Weighted_NIL_Worth', ascending=False).head(20)
print(top20[['Player','Team','Class','TalentScore','AgeFactor','Weighted_NIL_Worth']])


# In[123]:


import requests, pprint

# Pick a sample team
sample_tid, sample_name = team_list[0]
url = (
    "https://site.web.api.espn.com/apis/common/v3/sports/"
    "basketball/mens-college-basketball/teams/"
    f"{sample_tid}/roster?region=us&lang=en"
)
j = requests.get(url, headers=headers).json()

# Show the first positionGroup and its athlete structure
print("PositionGroups keys:", list(j['positionGroups'][0].keys()))
ath = j['positionGroups'][0].get('athletes', [None])[0]
print("Sample athlete keys:", list(ath.keys()) if ath else "No athletes here")
pprint.pprint(ath, width=100)


# In[131]:


# After df_roster = pd.concat(roster_dfs, ignore_index=True)
replacements = {
    'Middle Tennessee Blue': 'Middle Tennessee Blue Raiders',
    'Wright State': 'Wright State Raiders',
    'Colgate': 'Colgate Raiders',
    'Texas Tech Red': 'Texas Tech Red Raiders'
}
df_roster['Team'] = df_roster['Team'].replace(replacements)


# In[133]:


import requests
import pandas as pd

headers = {"User-Agent": "Mozilla/5.0"}

def fetch_espn_roster_json(team_id, team_name):
    """
    Use ESPN’s JSON API to fetch roster and pull class from the experience field.
    """
    url = (
        "https://site.web.api.espn.com/apis/common/v3/sports/"
        "basketball/mens-college-basketball/teams/"
        f"{team_id}/roster?region=us&lang=en"
    )
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    j = r.json()

    out = []
    for group in j.get("positionGroups", []):
        for athlete in group.get("athletes", []):
            name = athlete.get("displayName")
            exp  = athlete.get("experience", {}) or {}
            cls  = exp.get("displayValue", "Unknown")
            out.append({"Player": name, "Team": team_name, "Class": cls})
    return pd.DataFrame(out)


# Build the full roster DF —
roster_dfs = []
skipped = []
for tid, tname in team_list:
    try:
        df_r = fetch_espn_roster_json(tid, tname)
        roster_dfs.append(df_r)
    except Exception as e:
        skipped.append((tname, str(e)))

print(f"✅ Fetched rosters for {len(roster_dfs)} teams, skipped {len(skipped)} teams.")
df_roster = pd.concat(roster_dfs, ignore_index=True)
print("Roster columns:", df_roster.columns.tolist())
print("Unique teams:", df_roster['Team'].nunique())


# Merge into your stats DF —
df = pd.merge(df_stats, df_roster, on=['Player','Team'], how='left')
missing = df['Class'].isna().sum()
print(f"Players missing Class after merge: {missing}")
df['Class'] = df['Class'].fillna('Unknown')


# Age‑factor mapping & weighted NIL —
def age_factor(c):
    c = str(c).lower()
    if 'fresh' in c:    return 1.2
    if 'soph' in c:     return 1.15
    if 'junior' in c:   return 1.1
    if 'senior' in c:   return 1.05
    return 1.0

df['AgeFactor'] = df['Class'].apply(age_factor)
df['Weighted_NIL_Worth'] = (df['NIL_Worth'] * df['AgeFactor']).round().astype(int)


# Inspect top 20 —
top20 = df.sort_values('Weighted_NIL_Worth', ascending=False).head(20)
print(top20[['Player','Team','Class','TalentScore','AgeFactor','Weighted_NIL_Worth']])


# In[135]:


# Which teams appear in stats but not in roster?
stats_teams  = set(df_stats['Team'].unique())
roster_teams = set(df_roster['Team'].unique())
print("Missing in roster:", stats_teams - roster_teams)
print("Extra in roster:", roster_teams - stats_teams)


# In[137]:


# --- (Re)compute TalentScore & NIL_Worth ---

# 1) z‐score your features
features = ['PTS','AST','REB','STL','BLK','TO','FG%','3P%','FT%','WinPct']
for f in features:
    df[f] = pd.to_numeric(df[f], errors='coerce')
z = (df[features] - df[features].mean()) / df[features].std()
z['TO'] = -z['TO']

# 2) TalentScore
df['TalentScore'] = z.sum(axis=1)

# 3) Linear mapping to NIL_Worth using your two anchors
s_toppin = df.loc[df['Player'].str.contains('Toppin', case=False), 'TalentScore'].iat[0]
s_agbim   = df.loc[df['Player'].str.contains('Agbim',   case=False), 'TalentScore'].iat[0]
v_toppin, v_agbim = 4_000_000, 1_000_000
a = (v_toppin - v_agbim) / (s_toppin - s_agbim)
b = v_toppin - a * s_toppin
df['NIL_Worth'] = (df['TalentScore'] * a + b).clip(0, v_toppin).round().astype(int)


# In[138]:


def age_factor(c):
    cl = str(c).lower()
    if 'fresh' in cl:  return 1.2
    if 'soph' in cl:   return 1.15
    if 'junior' in cl: return 1.1
    if 'senior' in cl: return 1.05
    return 1.0

df['AgeFactor'] = df['Class'].apply(age_factor)
df['Weighted_NIL_Worth'] = (df['NIL_Worth'] * df['AgeFactor']).round().astype(int)

# Check top prospects
top20 = df.sort_values('Weighted_NIL_Worth', ascending=False).head(20)
print(top20[['Player','Team','Class','TalentScore','NIL_Worth','AgeFactor','Weighted_NIL_Worth']])


# In[139]:


# Show a handful of the players we couldn’t merge a Class for:
missing = df[df['Class']=='Unknown'][['Player','Team']].drop_duplicates().head(20)
print("Sample missing Player+Team combinations:")
print(missing.to_string(index=False))


# In[140]:


# --- 1) Build & clean roster DataFrame ---
roster_dfs = []
skipped = []
for tid, tname in team_list:
    try:
        df_r = fetch_espn_roster_json(tid, tname)
        roster_dfs.append(df_r)
    except Exception as e:
        skipped.append((tname, str(e)))

df_roster = pd.concat(roster_dfs, ignore_index=True)

# Drop the “Total” aggregate row if present
df_roster = df_roster[df_roster['Player'] != 'Total']

# Fix those four short‑name vs full‑name mismatches:
team_map = {
    'Middle Tennessee Blue':     'Middle Tennessee Blue Raiders',
    'Wright State':              'Wright State Raiders',
    'Colgate':                   'Colgate Raiders',
    'Texas Tech Red':            'Texas Tech Red Raiders'
}
df_roster['Team'] = df_roster['Team'].replace(team_map)

# --- 2) Normalize stats names to drop trailing position letter ---
# (Your df_stats Player column currently has names like "Johni Broome F")
df_stats['Player'] = df_stats['Player'].str.replace(r'\s+[FGC]$', '', regex=True)

# --- 3) Merge roster Class into stats ---
df = pd.merge(df_stats, df_roster, on=['Player','Team'], how='left')
missing = df['Class'].isna().sum()
print(f"Players missing Class after merge: {missing}")
df['Class'] = df['Class'].fillna('Unknown')

# --- 4) (Re)compute TalentScore & NIL_Worth, then apply age factor ---
# — z‑score features —
features = ['PTS','AST','REB','STL','BLK','TO','FG%','3P%','FT%','WinPct']
for f in features:
    df[f] = pd.to_numeric(df[f], errors='coerce')
z = (df[features] - df[features].mean()) / df[features].std()
z['TO'] = -z['TO']
df['TalentScore'] = z.sum(axis=1)

# — map TalentScore to NIL_Worth using your anchors —
s1 = df.loc[df['Player'].str.contains('Toppin',   case=False), 'TalentScore'].iat[0]
s2 = df.loc[df['Player'].str.contains('Agbim',    case=False), 'TalentScore'].iat[0]
v1, v2 = 4_000_000, 1_000_000
a = (v1 - v2) / (s1 - s2)
b = v1 - a * s1
df['NIL_Worth'] = (df['TalentScore'] * a + b).clip(0, v1).round().astype(int)

# — age factor & weighted NIL —
def age_factor(c):
    c = str(c).lower()
    if 'fresh'  in c: return 1.2
    if 'soph'   in c: return 1.15
    if 'junior' in c: return 1.1
    if 'senior' in c: return 1.05
    return 1.0

df['AgeFactor'] = df['Class'].apply(age_factor)
df['Weighted_NIL_Worth'] = (df['NIL_Worth'] * df['AgeFactor']).round().astype(int)

# --- 5) Inspect the top 20 by weighted NIL worth ---
top20 = df.sort_values('Weighted_NIL_Worth', ascending=False).head(20)
print(top20[['Player','Team','Class','TalentScore','NIL_Worth','AgeFactor','Weighted_NIL_Worth']])


# In[145]:


# make sure the folder exists
import os
os.makedirs('data', exist_ok=True)

# export for the Next.js API to consume
df.to_json('data/players.json', orient='records')
print("✅ Wrote data/players.json with", len(df), "records")


# In[144]:


import os
print(os.getcwd())


# In[160]:


df.to_json('player.json', orient='records')


# In[161]:


import os

# adjust this to the absolute path of your Next project
out_path = os.path.expanduser('~/projects/nil-scout-frontend/data/players.json')
df.to_json(out_path, orient='records')
print("Wrote", out_path)


# In[162]:


# adjust this path to wherever your Next project lives:
output_path = '/Users/aaronpearlstein/projects/nil-scout-frontend/data/players.json'
df.to_json(output_path, orient='records')
print("Wrote", len(df), "records to", output_path)


# In[163]:


cp /path/to/your/generated/players.json ~/projects/nil-scout-frontend/data/players.json


# In[164]:


# app.py
import streamlit as st
import pandas as pd

@st.cache
def load_data():
    return pd.read_json("players.json")

df = load_data()

st.title("🏀 NIL Scout Dashboard")
search = st.text_input("Search player or team…")
filtered = df[df["Player"].str.contains(search, case=False) |
              df["Team"].str.contains(search, case=False)]

st.dataframe(filtered)

