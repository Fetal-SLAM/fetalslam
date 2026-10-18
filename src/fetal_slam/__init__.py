"""Maintained offline fetal SLAM runtime."""

__all__ = ["XFeatSlamOffline"]


def __getattr__(name):
    if name == "XFeatSlamOffline":
        from .slam import XFeatSlamOffline

        return XFeatSlamOffline
    raise AttributeError(name)
