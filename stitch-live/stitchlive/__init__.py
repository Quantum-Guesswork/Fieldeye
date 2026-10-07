"""stitchlive - skladanie zdjec z drona w czasie rzeczywistym na bazie OpenCV."""

__version__ = "0.1.0"

from .basic import SkladaczPodstawowy
from .detail import SkladaczDetail
from .globalny import SkladaczGlobalny
from .source import ZrodloKatalog, ZrodloWsadowe

__all__ = ["SkladaczPodstawowy", "SkladaczDetail", "SkladaczGlobalny",
           "ZrodloKatalog", "ZrodloWsadowe"]
