"""Audit trail for golden-path actions.

Every mutating golden-path request records one append-only audit event here
(see `events.py`), attributed to the actor and the resource it touched. The
mock observer audit router reads these on top of its synthetic baseline, so the
Audit Logs page reflects real platform activity instead of canned rows.
"""
