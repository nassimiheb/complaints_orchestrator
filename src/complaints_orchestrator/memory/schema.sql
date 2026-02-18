CREATE TABLE IF NOT EXISTS customers_memory (
    customer_id TEXT PRIMARY KEY,
    preferred_language TEXT NOT NULL,
    ninety_day_compensation_total REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cases_memory (
    case_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    status TEXT NOT NULL,
    compensation_value REAL NOT NULL DEFAULT 0,
    opened_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_memory_customer_id ON cases_memory(customer_id);
CREATE INDEX IF NOT EXISTS idx_cases_memory_opened_at ON cases_memory(opened_at);
