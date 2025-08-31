import os
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

EXTERNAL_DIR = Path("data/raw/external")
CACHE_DIR = EXTERNAL_DIR / "cache"
STATION = "HKO"

URL_TEMP = f"https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMTEMP&rformat=csv&station={STATION}"
URL_RAIN_ALL = f"https://data.weather.gov.hk/weatherAPI/cis/csvfile/{STATION}/ALL/daily_{STATION}_RF_ALL.csv"

def session_with_retries():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "hk-weather-fetch/1.0"})
    return s

def fetch_csv(url: str, cache_name: str) -> pd.DataFrame:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s = session_with_retries()
    try:
        r = s.get(url, timeout=30)
        r.raise_for_status()
        # write a cache copy so you can re-run offline
        cache_path = CACHE_DIR / cache_name
        cache_path.write_bytes(r.content)
        return pd.read_csv(cache_path)
    except Exception as e:
        # Fallback to cache if present
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            print(f"[warn] Network failed, using cached file: {cache_path.name} ({e})")
            return pd.read_csv(cache_path)
        raise RuntimeError(
            f"Download failed for {url} ({e}). "
            f"If you are behind a proxy, set HTTPS_PROXY/HTTP_PROXY. "
            f"Or manually download the file to: {cache_path}"
        )

def rename_mixed_cols(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    colmap = {}
    for c in df.columns:
        for key, val in mapping.items():
            if key in c:
                colmap[c] = val
                break
    return df.rename(columns=colmap)

def add_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        date=pd.to_datetime(
            dict(year=df["Year"], month=df["Month"], day=df["Day"]),
            errors="coerce"
        )
    ).dropna(subset=["date"])

def tidy_temperature(df_temp: pd.DataFrame) -> pd.DataFrame:
    # Normalize likely columns
    cm = rename_mixed_cols(
        df_temp,
        {"Year": "Year", "Month": "Month", "Day": "Day",
         "Tmax": "Tmax", "Tmin": "Tmin", "Tmean": "Tmean",
         "Maximum": "Tmax", "Minimum": "Tmin", "Mean": "Tmean"}
    )
    cols = [c for c in ["Year","Month","Day","Tmax","Tmin","Tmean"] if c in cm.columns]
    df = cm[cols].copy()
    df = add_date(df)
    # Convert numerics safely
    for c in ["Tmax","Tmin","Tmean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def tidy_rainfall(df_rf: pd.DataFrame) -> pd.DataFrame:
    df = rename_mixed_cols(
        df_rf,
        {"Year":"Year","Month":"Month","Day":"Day",
         "Value":"Rain_mm", "數值":"Rain_mm",
         "Completeness":"Completeness","完整":"Completeness"}
    )
    cols = [c for c in ["Year","Month","Day","Rain_mm"] if c in df.columns]
    df = df[cols].copy()
    df = add_date(df)
    df["Rain_mm"] = pd.to_numeric(df["Rain_mm"], errors="coerce")
    return df

def merge_temp_rain(temp: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(temp, rain[["date","Rain_mm"]], on="date", how="outer").sort_values("date")

def slice_year_month(df: pd.DataFrame, year: int, month: int | None = None) -> pd.DataFrame:
    df = df[df["date"].dt.year == year]
    if month:
        df = df[df["date"].dt.month == month]
    return df

def main():
    # Try to download (or use cache)
    temp_csv = fetch_csv(URL_TEMP, f"{STATION}_CLMTEMP_all.csv")
    rain_csv = fetch_csv(URL_RAIN_ALL, f"{STATION}_RF_ALL.csv")

    temp = tidy_temperature(temp_csv)
    rain = tidy_rainfall(rain_csv)
    combined = merge_temp_rain(temp, rain)

    # Outputs required by the challenge
    y2023 = slice_year_month(combined, 2023)
    (EXTERNAL_DIR / "hk_weather_2023.csv").write_text(y2023.to_csv(index=False), encoding="utf-8")

    jan2024 = slice_year_month(combined, 2024, 1)
    (EXTERNAL_DIR / "hk_weather_2024_jan.csv").write_text(jan2024.to_csv(index=False), encoding="utf-8")

    print("Saved:")
    print(" -", (EXTERNAL_DIR / "hk_weather_2023.csv").as_posix())
    print(" -", (EXTERNAL_DIR / "hk_weather_2024_jan.csv").as_posix())
    print("\nIf you hit DNS/proxy errors again, manually download these two:")
    print(" 1) ", URL_TEMP, "  -> cache file:", (CACHE_DIR / f"{STATION}_CLMTEMP_all.csv").as_posix())
    print(" 2) ", URL_RAIN_ALL, " -> cache file:", (CACHE_DIR / f"{STATION}_RF_ALL.csv").as_posix())

if __name__ == "__main__":
    main()
