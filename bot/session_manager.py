"""Session manager for rate limiting and user tracking"""

import time
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

from .exceptions import RateLimitExceededError


class UserSession:
    """Track individual user session"""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.request_timestamps: List[float] = []
        self.total_requests = 0
        self.total_cost = 0.0
        self.total_entities_extracted = 0
        self.first_request_time = None
        self.last_request_time = None

    def add_request(self, cost: float = 0.0, entities: int = 0):
        """Record a new request"""
        now = time.time()
        self.request_timestamps.append(now)
        self.total_requests += 1
        self.total_cost += cost
        self.total_entities_extracted += entities
        self.last_request_time = now

        if self.first_request_time is None:
            self.first_request_time = now

    def get_requests_in_last_hour(self) -> int:
        """Get number of requests in last hour"""
        one_hour_ago = time.time() - 3600
        # Clean up old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if ts > one_hour_ago
        ]
        return len(self.request_timestamps)

    def can_make_request(self, max_per_hour: int) -> bool:
        """Check if user can make another request"""
        return self.get_requests_in_last_hour() < max_per_hour

    def get_stats(self) -> Dict:
        """Get user statistics"""
        return {
            "user_id": self.user_id,
            "total_requests": self.total_requests,
            "total_cost": self.total_cost,
            "total_entities": self.total_entities_extracted,
            "requests_last_hour": self.get_requests_in_last_hour(),
            "first_request": datetime.fromtimestamp(self.first_request_time).isoformat() if self.first_request_time else None,
            "last_request": datetime.fromtimestamp(self.last_request_time).isoformat() if self.last_request_time else None
        }


class SessionManager:
    """Manage user sessions and rate limiting"""

    def __init__(self, max_requests_per_hour: int = 5):
        self.max_requests_per_hour = max_requests_per_hour
        self.sessions: Dict[int, UserSession] = {}
        self.active_processing: Dict[int, bool] = {}  # Track who's currently processing

    def get_or_create_session(self, user_id: int) -> UserSession:
        """Get existing session or create new one"""
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id)
        return self.sessions[user_id]

    def check_rate_limit(self, user_id: int) -> bool:
        """
        Check if user can make a request

        Args:
            user_id: Telegram user ID

        Returns:
            True if request allowed

        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        session = self.get_or_create_session(user_id)

        if not session.can_make_request(self.max_requests_per_hour):
            raise RateLimitExceededError(
                user_id=user_id,
                limit=self.max_requests_per_hour,
                time_window="hour"
            )

        return True

    def start_processing(self, user_id: int) -> bool:
        """
        Mark user as currently processing

        Args:
            user_id: Telegram user ID

        Returns:
            True if started, False if already processing
        """
        if user_id in self.active_processing and self.active_processing[user_id]:
            return False  # Already processing

        self.active_processing[user_id] = True
        return True

    def finish_processing(self, user_id: int, cost: float = 0.0, entities: int = 0):
        """
        Mark processing as complete and update stats

        Args:
            user_id: Telegram user ID
            cost: Processing cost in USD
            entities: Number of entities extracted
        """
        session = self.get_or_create_session(user_id)
        session.add_request(cost=cost, entities=entities)

        if user_id in self.active_processing:
            self.active_processing[user_id] = False

    def is_processing(self, user_id: int) -> bool:
        """Check if user is currently processing a request"""
        return self.active_processing.get(user_id, False)

    def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for specific user"""
        session = self.get_or_create_session(user_id)
        return session.get_stats()

    def get_global_stats(self) -> Dict:
        """Get global statistics across all users"""
        total_users = len(self.sessions)
        total_requests = sum(s.total_requests for s in self.sessions.values())
        total_cost = sum(s.total_cost for s in self.sessions.values())
        total_entities = sum(s.total_entities_extracted for s in self.sessions.values())
        active_now = sum(1 for is_active in self.active_processing.values() if is_active)

        return {
            "total_users": total_users,
            "total_requests": total_requests,
            "total_cost": total_cost,
            "total_entities": total_entities,
            "active_processing": active_now,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "avg_entities_per_request": total_entities / total_requests if total_requests > 0 else 0
        }

    def cleanup_old_sessions(self, max_age_days: int = 7):
        """
        Remove sessions older than max_age_days with no recent activity

        Args:
            max_age_days: Maximum age in days
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        users_to_remove = []

        for user_id, session in self.sessions.items():
            if session.last_request_time and session.last_request_time < cutoff_time:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del self.sessions[user_id]
            if user_id in self.active_processing:
                del self.active_processing[user_id]
