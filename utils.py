import logging
import os
import sys

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState, TryAgain, \
    before_log, retry_if_exception


# tenacity retry settings
retry_settings = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(
        lambda e: isinstance(e, requests.exceptions.HTTPError)
                  and e.response is not None
                  and (e.response.status_code == 429 or 500 <= e.response.status_code < 600)
    ),
    before=before_log(logging, logging.WARNING),
    reraise=True
)
