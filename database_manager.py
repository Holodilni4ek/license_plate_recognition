#!/usr/bin/env python3
"""
Database Manager
Handles database connections, connection pooling, and common database operations.
"""

import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")


class DatabaseManager:
    """Manages database connections and provides common database operations."""

    def __init__(self):
        """Initialize database manager with connection pool."""
        self.connection_pool = None
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        # Validate configuration
        if not all(self.db_config.values()):
            missing_vars = [k for k, v in self.db_config.items() if not v]
            raise ValueError(
                f"Missing database configuration variables: {missing_vars}"
            )

        self._create_connection_pool()

    def _create_connection_pool(self):
        """Create a connection pool for database connections."""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20, **self.db_config  # Min and max connections
            )
            logging.info("Database connection pool created successfully")
        except Exception as e:
            logging.error(f"Failed to create database connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logging.error(f"Database operation failed: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)

    def execute_query(self, query: str, params: Tuple = None) -> List[Tuple]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchall()

    def execute_insert(self, query: str, params: Tuple = None) -> Optional[Any]:
        """Execute an INSERT query and return the inserted ID if available."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                conn.commit()
                try:
                    return cursor.fetchone()[0]  # Return inserted ID
                except (TypeError, IndexError):
                    return None

    def execute_update(self, query: str, params: Tuple = None) -> int:
        """Execute an UPDATE/DELETE query and return number of affected rows."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                conn.commit()
                return cursor.rowcount

    def get_log_entries(self, date: str, limit: int = 50) -> List[Tuple]:
        """Get log entries for a specific date."""
        query = """
        SELECT
            CONCAT_WS(' ', v.vehicle_color, v.vehicle_type, v.vehicle_mark) AS vehicle,
            CONCAT_WS(' ', d.driver_firstname, d.driver_secondname, d.driver_patronymic) AS driver,
            l.transittime,
            CASE WHEN transittype THEN 'Заехал' 
            ELSE 'Выехал' 
            END AS status
        FROM "log" AS l
        JOIN "vehicle" AS v ON l.id_vehicle = v.id_vehicle
        JOIN "driver" AS d ON v.id_driver = d.id_driver
        WHERE l.transittime::date = %s
        ORDER BY l.transittime DESC
        LIMIT %s
        """
        return self.execute_query(query, (date, limit))

    def add_log_entry(self, vehicle_number: str) -> bool:
        """Add a new log entry for a vehicle."""
        query = """
        WITH get_vehicle AS (
            SELECT id_vehicle 
            FROM vehicle 
            WHERE vehicle_mark = %s
            LIMIT 1
        )
        INSERT INTO log (
            id_vehicle, 
            transittime, 
            transittype
        )
        SELECT 
            id_vehicle,
            DATE_TRUNC('second', CURRENT_TIMESTAMP),
            TRUE
        FROM get_vehicle
        RETURNING id_log
        """
        try:
            result = self.execute_insert(query, (vehicle_number,))
            return result is not None
        except Exception as e:
            logging.error(f"Failed to add log entry for vehicle {vehicle_number}: {e}")
            return False

    def is_vehicle_registered(self, vehicle_number: str) -> bool:
        """Check if a vehicle number is registered."""
        query = "SELECT 1 FROM vehicle WHERE vehicle_mark = %s LIMIT 1"
        result = self.execute_query(query, (vehicle_number,))
        return len(result) > 0

    def get_registered_vehicles(self) -> set:
        """Get all registered vehicle numbers."""
        query = "SELECT vehicle_mark FROM vehicle"
        result = self.execute_query(query)
        return {row[0] for row in result}

    def add_driver(
        self,
        firstname: str,
        secondname: str,
        patronymic: str,
        birthdate: str,
        nationality: str,
    ) -> Optional[int]:
        """Add a new driver to the database."""
        query = """
        INSERT INTO driver (driver_firstname, driver_secondname, driver_patronymic, 
                           driver_birthdate, driver_nationality) 
        VALUES (%s, %s, %s, %s, %s) 
        RETURNING id_driver
        """
        try:
            return self.execute_insert(
                query, (firstname, secondname, patronymic, birthdate, nationality)
            )
        except Exception as e:
            logging.error(f"Failed to add driver: {e}")
            return None

    def add_vehicle(
        self, vehicle_mark: str, vehicle_color: str, vehicle_type: str, driver_id: int
    ) -> Optional[int]:
        """Add a new vehicle to the database."""
        query = """
        INSERT INTO vehicle (vehicle_mark, vehicle_color, vehicle_type, id_driver) 
        VALUES (%s, %s, %s, %s) 
        RETURNING id_vehicle
        """
        try:
            return self.execute_insert(
                query, (vehicle_mark, vehicle_color, vehicle_type, driver_id)
            )
        except Exception as e:
            logging.error(f"Failed to add vehicle: {e}")
            return None

    def add_user(self, login: str, password: str) -> Optional[int]:
        """Add a new user account to the database."""
        import hashlib

        # Hash the credentials
        login_hash = hashlib.sha256(login.encode("utf-8")).hexdigest()
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        query = """
        INSERT INTO private.account (login, password) 
        VALUES (%s, %s) 
        RETURNING login
        """
        try:
            return self.execute_insert(query, (login_hash, password_hash))
        except Exception as e:
            logging.error(f"Failed to add user: {e}")
            return None

    def authenticate_user(self, login: str, password: str) -> bool:
        """Authenticate a user login."""
        import hashlib

        login_hash = hashlib.sha256(login.encode("utf-8")).hexdigest()
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        query = """
        SELECT login, password FROM private.account 
        WHERE login = %s 
        LIMIT 1
        """
        try:
            result = self.execute_query(query, (login_hash,))
            if result and len(result) > 0:
                stored_login, stored_password = result[0]
                return stored_login == login_hash and stored_password == password_hash
            return False
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            return False

    def close(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logging.info("Database connection pool closed")


# Global database manager instance
db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager
