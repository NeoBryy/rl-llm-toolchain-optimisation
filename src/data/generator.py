"""
Synthetic utilities meter data generator for RL Meter Analyst project.

Generates realistic time-series consumption data with:
- Customer demographics (residential/commercial segments)
- Tariff structures
- 15-minute interval meter readings with realistic temporal patterns

Usage:
    from src.data import generator
    from src import config

    generator.create_database(config.Paths.DB_PATH, overwrite=True)
"""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_customers(num_customers: int, seed: int) -> pd.DataFrame:
    """
    Generate customer records with segments and meter types.

    Args:
        num_customers: Number of customer records to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: customer_id, segment, meter_type, join_date
    """
    rng = np.random.default_rng(seed)

    # 70% residential, 30% commercial
    segments = rng.choice(
        config.DataConfig.CUSTOMER_SEGMENTS, size=num_customers, p=[0.7, 0.3]
    )

    # 80% smart meters, 20% standard
    meter_types = rng.choice(
        config.DataConfig.METER_TYPES, size=num_customers, p=[0.8, 0.2]
    )

    # Join dates spread over past 5 years
    base_date = datetime.now() - timedelta(days=5 * 365)
    join_dates = [
        base_date + timedelta(days=int(rng.integers(0, 5 * 365)))
        for _ in range(num_customers)
    ]

    customers_df = pd.DataFrame({
        "customer_id": range(1, num_customers + 1),
        "segment": segments,
        "meter_type": meter_types,
        "join_date": join_dates,
    })

    logger.info(
        "Generated %d customers (%d residential, %d commercial)",
        num_customers,
        (segments == "residential").sum(),
        (segments == "commercial").sum(),
    )

    return customers_df


def generate_tariffs() -> pd.DataFrame:
    """
    Generate tariff structures for customer segments with time-of-use pricing.

    Returns:
        DataFrame with columns: tariff_id, segment, peak_rate_per_kwh,
                                off_peak_rate_per_kwh, standing_charge, peak_hours
    """
    tariffs_df = pd.DataFrame([
        {
            "tariff_id": 1,
            "segment": "residential",
            "peak_rate_per_kwh": 0.35,  # £0.35 per kWh during peak (higher)
            "off_peak_rate_per_kwh": 0.22,  # £0.22 per kWh off-peak (lower)
            "standing_charge": 0.45,  # £0.45 per day
            "peak_hours": "06:00-09:00,18:00-22:00",
        },
        {
            "tariff_id": 2,
            "segment": "commercial",
            "peak_rate_per_kwh": 0.28,  # £0.28 per kWh during peak
            "off_peak_rate_per_kwh": 0.18,  # £0.18 per kWh off-peak
            "standing_charge": 1.20,  # £1.20 per day (higher fixed cost)
            "peak_hours": "09:00-17:00",
        },
    ])

    logger.info("Generated %d tariff structures", len(tariffs_df))

    return tariffs_df


def _generate_timestamp_range(
    start_date: datetime, days: int, interval_minutes: int
) -> list[datetime]:
    """
    Generate list of timestamps at specified intervals.

    Args:
        start_date: Starting datetime
        days: Number of days to generate
        interval_minutes: Minutes between each timestamp

    Returns:
        List of datetime objects
    """
    intervals_per_day = (24 * 60) // interval_minutes
    return [
        start_date + timedelta(minutes=interval_minutes * i)
        for i in range(days * intervals_per_day)
    ]


def _calculate_base_consumption(segment: str, hour: int, is_weekday: bool) -> float:
    """
    Calculate base consumption for segment and time of day.

    Patterns:
    - Residential: peak 6-9am (0.8 kWh), 6-10pm (1.0 kWh), off-peak (0.3 kWh)
    - Commercial: peak 9am-5pm weekdays (2.5 kWh), off-peak/weekends (0.8 kWh)

    Args:
        segment: Customer segment (residential/commercial)
        hour: Hour of day (0-23)
        is_weekday: Whether timestamp is a weekday

    Returns:
        Base consumption in kWh for 15-minute interval
    """
    if segment == "residential":
        if 6 <= hour < 9 or 18 <= hour < 22:
            return 0.8 if 6 <= hour < 9 else 1.0
        return 0.3
    else:  # commercial
        if is_weekday and 9 <= hour < 17:
            return 2.5
        return 0.8


def _apply_seasonal_factor(month: int) -> float:
    """
    Return seasonal multiplier based on month.

    - Winter (Dec-Feb): 1.3x (heating)
    - Summer (Jun-Aug): 1.2x (cooling)
    - Spring/Fall: 0.85x (mild weather)

    Args:
        month: Month number (1-12)

    Returns:
        Seasonal multiplier
    """
    if month in [12, 1, 2]:  # Winter
        return 1.3
    elif month in [6, 7, 8]:  # Summer
        return 1.2
    return 0.85  # Spring/Fall


def _create_reading_record(
    customer_id: int,
    segment: str,
    timestamp: datetime,
    reading_id: int,
    rng: np.random.Generator,
) -> dict:
    """
    Create single reading record with realistic consumption.

    Args:
        customer_id: Customer identifier
        segment: Customer segment (residential/commercial)
        timestamp: Reading timestamp
        reading_id: Unique reading identifier
        rng: NumPy random generator for reproducibility

    Returns:
        Dictionary with reading_id, customer_id, timestamp, consumption_kwh
    """
    hour = timestamp.hour
    month = timestamp.month
    is_weekday = timestamp.weekday() < 5

    base = _calculate_base_consumption(segment, hour, is_weekday)
    seasonal = _apply_seasonal_factor(month)
    mean = base * seasonal
    consumption = max(0, rng.normal(mean, mean * 0.15))

    return {
        "reading_id": reading_id,
        "customer_id": customer_id,
        "timestamp": timestamp,
        "consumption_kwh": round(consumption, 3),
    }


def generate_readings(
    customers: pd.DataFrame,
    date_range_days: int,
    interval_minutes: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate time-series meter readings with realistic patterns.

    Uses helper functions to calculate consumption based on:
    - Time of day and customer segment
    - Seasonal variations
    - Random noise

    Args:
        customers: DataFrame of customer records
        date_range_days: Number of days to generate readings for
        interval_minutes: Minutes between readings (e.g., 15)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: reading_id, customer_id, timestamp, consumption_kwh
    """
    rng = np.random.default_rng(seed)

    # Generate timestamp range
    start_date = datetime.now() - timedelta(days=date_range_days)
    timestamps = _generate_timestamp_range(start_date, date_range_days, interval_minutes)

    intervals_per_day = (24 * 60) // interval_minutes
    total_readings = len(customers) * date_range_days * intervals_per_day

    logger.info(
        "Generating %d total readings (%d customers × %d days × %d intervals/day)",
        total_readings,
        len(customers),
        date_range_days,
        intervals_per_day,
    )

    readings_list = []
    reading_id = 1

    for idx, customer in customers.iterrows():
        for timestamp in timestamps:
            reading = _create_reading_record(
                customer["customer_id"], customer["segment"], timestamp, reading_id, rng
            )
            readings_list.append(reading)
            reading_id += 1

        if (idx + 1) % 20 == 0:
            logger.debug("Generated readings for %d/%d customers", idx + 1, len(customers))

    readings_df = pd.DataFrame(readings_list)
    logger.info(
        "Generated %d readings (%.2f MB in memory)",
        len(readings_df),
        readings_df.memory_usage(deep=True).sum() / 1024 / 1024,
    )

    return readings_df


def _write_to_database(
    db_path: Path,
    customers_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    readings_df: pd.DataFrame,
) -> None:
    """
    Write DataFrames to DuckDB with indexes.

    Args:
        db_path: Path to DuckDB database file
        customers_df: Customer records DataFrame
        tariffs_df: Tariff structures DataFrame
        readings_df: Meter readings DataFrame

    Raises:
        duckdb.Error: If database operations fail
    """
    logger.info("Writing data to DuckDB at %s", db_path)

    try:
        with duckdb.connect(str(db_path)) as conn:
            # Create tables
            conn.execute("CREATE TABLE customers AS SELECT * FROM customers_df")
            conn.execute("CREATE TABLE tariffs AS SELECT * FROM tariffs_df")
            conn.execute("CREATE TABLE readings AS SELECT * FROM readings_df")

            # Create indexes for query performance
            conn.execute("CREATE INDEX idx_customer_id ON customers(customer_id)")
            conn.execute("CREATE INDEX idx_readings_customer ON readings(customer_id)")
            conn.execute("CREATE INDEX idx_readings_timestamp ON readings(timestamp)")

            # Log table stats
            for table_name in config.DataConfig.TABLE_NAMES:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                logger.info("Table '%s': %d rows", table_name, count)

        logger.info("Database created successfully at %s", db_path)

    except duckdb.Error as e:
        logger.error("Database creation failed: %s", e)
        raise


def create_database(db_path: Path, overwrite: bool = False) -> None:
    """
    Main entry point: generate all tables and write to DuckDB.

    Creates three tables:
    - customers: Customer demographics and meter types
    - tariffs: Pricing structures by segment
    - readings: Time-series consumption data

    Args:
        db_path: Path to DuckDB database file
        overwrite: If True, regenerate data even if database exists

    Raises:
        ValueError: If configuration is invalid
        duckdb.Error: If database operations fail
    """
    # Input validation
    if config.DataConfig.NUM_CUSTOMERS <= 0:
        raise ValueError("NUM_CUSTOMERS must be positive")
    if config.DataConfig.DATE_RANGE_DAYS <= 0:
        raise ValueError("DATE_RANGE_DAYS must be positive")
    if config.DataConfig.READING_INTERVAL_MINUTES <= 0:
        raise ValueError("READING_INTERVAL_MINUTES must be positive")

    # Check if database exists
    if db_path.exists() and not overwrite:
        logger.warning("Database exists at %s. Use overwrite=True to regenerate.", db_path)
        return

    logger.info("Creating database at %s", db_path)

    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing database")

    # Generate data
    logger.info("Generating synthetic data with seed=%d", config.DataConfig.RANDOM_SEED)

    customers_df = generate_customers(
        config.DataConfig.NUM_CUSTOMERS, config.DataConfig.RANDOM_SEED
    )
    tariffs_df = generate_tariffs()
    readings_df = generate_readings(
        customers_df,
        config.DataConfig.DATE_RANGE_DAYS,
        config.DataConfig.READING_INTERVAL_MINUTES,
        config.DataConfig.RANDOM_SEED,
    )

    # Write to database
    _write_to_database(db_path, customers_df, tariffs_df, readings_df)
