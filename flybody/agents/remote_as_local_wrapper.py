"""Wrapper to call methods of remote Ray actors as if they were local."""

import inspect
import ray


class RemoteAsLocal():
    """This wrappers allows calling methods of remote Ray actors (e.g. classes
    decorated with @ray.remote) as if they were local. It can be used to wrap
    classes from external libraries to simplify their integration with Ray.

    Example:

    @ray.remote
    class Counter():
        def __init__(self):
            self._counts = 0
        def increment(self, inc=1):
            self._counts += inc
        def get_counts(self):
            return self._counts

    # Normal Ray usage (without this wrapper):
    counter = Counter.remote()  # Instantiate as remote.
    counter.increment.remote(inc=2)  # Call as remote.
    obj_ref = counter.get_counts.remote()  # Call as remote; returns a future.
    ray.get(obj_ref)  # Blocks and returns 2.

    # Using Ray with this wrapper:
    counter = Counter.remote()  # Instantiate as remote.
    counter = RemoteAsLocal(counter)  # Wrap.
    counter.increment(inc=2)  # Call as local.

    # Can be called to either return a future or block until call returns (the
    # latter is the default behavior):
    obj_ref = counter.get_counts(block=False)  # Call as local; returns a future.
    counter.get_counts(block=True)  # Call as local; blocks and returns 2.
    """

    def __init__(self, remote_handle):
        """
        Args:
            remote_handle: Remote Ray class handle to be wrapped, see the
                docstring above.
        """

        self._remote_handle = remote_handle

        def remote_caller(method_name):
            # Wrapper for remote class's methods to mimic local calling.
            def wrapper(*args, block=True, **kwargs):
                obj_ref = getattr(self._remote_handle,
                                  method_name).remote(*args, **kwargs)
                if block:
                    return ray.get(
                        obj_ref)  # Block until called method returns.
                else:
                    return obj_ref  # Don't block and return a future.

            return wrapper

        for member in inspect.getmembers(self._remote_handle):
            name = member[0]
            if not name.startswith('__'):
                # Wrap public methods for remote-as-local calls.
                setattr(self, name, remote_caller(name))
            else:
                # Reassign dunder members for API-unaware callers (e.g. pickle).
                # For example, it is doing the following reassignment:
                # self.__reduce__ = self._remote_handle.__reduce__
                setattr(self, name, getattr(self._remote_handle, name))

    def __dir__(self):
        return dir(self._remote_handle)
