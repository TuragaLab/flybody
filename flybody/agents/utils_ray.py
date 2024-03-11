"""Utilities for Ray."""

import re
import ray


def get_actor_id(actor_handle: ray.actor.ActorHandle) -> str:
    """Return Ray Actor's ID."""
    if hasattr(actor_handle, '_remote_handle'):
        # For compatibility with RemoteAsLocal wrapper.
        actor_handle = actor_handle._remote_handle
    return re.findall('[abcdef\d]+\)$', str(actor_handle))[0][:-1]


def is_alive(actor_handle: ray.actor.ActorHandle) -> bool:
    """Check if actor is alive."""
    actor_id = get_actor_id(actor_handle)
    return ray.state.actors(actor_id)['State'] == 'ALIVE'
