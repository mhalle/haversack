"""The portable exclusive lock both install paths take.

Both callers used to call ``fcntl`` directly and fall back to NO lock where it
could not be imported - which is Windows, so two installs of one model there
could still destroy each other's weights. These tests run the same assertions
over whichever facility the platform has.
"""
import os
import tempfile
import threading
import unittest

from haversack import filelock


class ExclusiveLock(unittest.TestCase):

    def _two_descriptors(self, td):
        path = os.path.join(td, "x.lock")
        a = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        b = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        return a, b

    def test_a_second_holder_is_refused_until_the_first_releases(self):
        if not filelock.SUPPORTED:
            self.skipTest("no lock facility on this platform")
        with tempfile.TemporaryDirectory() as td:
            a, b = self._two_descriptors(td)
            try:
                self.assertTrue(filelock.lock(a))
                self.assertFalse(filelock.lock(b, blocking=False))
                filelock.unlock(a)
                self.assertTrue(filelock.lock(b, blocking=False))
                filelock.unlock(b)
            finally:
                os.close(a)
                os.close(b)

    def test_blocking_waits_for_the_release(self):
        """The install lock waits out a multi-minute install; a blocking lock
        that gave up after a fixed number of retries (msvcrt's LK_LOCK does)
        would let the second installer in."""
        if not filelock.SUPPORTED:
            self.skipTest("no lock facility on this platform")
        with tempfile.TemporaryDirectory() as td:
            a, b = self._two_descriptors(td)
            try:
                self.assertTrue(filelock.lock(a))
                got = []

                def waiter():
                    got.append(filelock.lock(b, poll_s=0.01))

                t = threading.Thread(target=waiter)
                t.start()
                t.join(0.3)
                self.assertTrue(t.is_alive(), "the waiter got in while the lock was held")
                filelock.unlock(a)
                t.join(5)
                self.assertEqual(got, [True])
                filelock.unlock(b)
            finally:
                os.close(a)
                os.close(b)

    def test_closing_the_descriptor_releases(self):
        if not filelock.SUPPORTED:
            self.skipTest("no lock facility on this platform")
        with tempfile.TemporaryDirectory() as td:
            a, b = self._two_descriptors(td)
            self.assertTrue(filelock.lock(a))
            os.close(a)
            try:
                self.assertTrue(filelock.lock(b, blocking=False))
            finally:
                os.close(b)

    def test_the_install_lock_uses_it(self):
        """The ecosystem install lock must serialize: a second `ensure` of the
        same folder waits rather than running beside the first."""
        from haversack.ecosystems import ZipManifestEcosystem
        if not filelock.SUPPORTED:
            self.skipTest("no lock facility on this platform")

        class Eco(ZipManifestEcosystem):
            name, bucket, generator = "e", "e", "t"
            MANIFEST = None

            def __init__(self):
                self._entries = {"t": {"folder": "Dataset001", "url": "x"}}

        eco = Eco()
        with tempfile.TemporaryDirectory() as td:
            order = []
            inside = threading.Event()
            release = threading.Event()

            def first():
                with eco._install_lock("t", td):
                    order.append("first-in")
                    inside.set()
                    release.wait(5)
                    order.append("first-out")

            def second():
                inside.wait(5)
                with eco._install_lock("t", td):
                    order.append("second-in")

            t1, t2 = threading.Thread(target=first), threading.Thread(target=second)
            t1.start()
            t2.start()
            inside.wait(5)
            t2.join(0.3)
            self.assertTrue(t2.is_alive(), "the second install got in beside the first")
            release.set()
            t1.join(5)
            t2.join(5)
            self.assertEqual(order, ["first-in", "first-out", "second-in"])


if __name__ == "__main__":
    unittest.main()
