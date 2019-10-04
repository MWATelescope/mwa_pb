"""Library to access the MWA metadata web services
"""

import json

try:   # Python3
    from urllib.parse import urlencode
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError
except ImportError:   # Python2
    from urllib import urlencode
    from urllib2 import urlopen, HTTPError, URLError

# Append the service name to this base URL, eg 'con', 'obs', etc.
BASEURL = 'http://ws.mwatelescope.org/'


# Function to call a JSON web service and return a dictionary:

def getmeta(servicetype='metadata', service='obs', params=None):
    """Given a JSON web servicetype ('observation' or 'metadata'), a service name (eg 'obs', find, or 'con')
       and a set of parameters as a Python dictionary, return a Python dictionary containing the result.
    """
    if params:
        data = urlencode(params)  # Turn the dictionary into a string with encoded 'name=value' pairs
    else:
        data = ''
    # Get the data
    try:
        result = json.load(urlopen(BASEURL + servicetype + '/' + service + '?' + data))
    except HTTPError as error:
        print(("HTTP error from server: code=%d, response:\n %s" % (error.code, error.read())))
        return
    except URLError as error:
        print(("URL or network error: %s" % error.reason))
        return
    # Return the result dictionary
    return result


def get_observation(obsid=None):
    """Get an observation structure from the metadata web service, given an obsid.
    """
    if obsid is None:
        obs = getmeta(servicetype='metadata', service='obs')
    else:
        obs = getmeta(servicetype='metadata', service='obs', params={'obs_id': obsid})
    return obs
