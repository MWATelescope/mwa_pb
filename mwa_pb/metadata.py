
"""Library to access the MWA metadata web services
"""

import urllib
import urllib2
import json

# Append the service name to this base URL, eg 'con', 'obs', etc.
BASEURL = 'http://mwa-metadata01.pawsey.org.au/'


# Function to call a JSON web service and return a dictionary:

def getmeta(servicetype='metadata', service='obs', params=None):
  """Given a JSON web servicetype ('observation' or 'metadata'), a service name (eg 'obs', find, or 'con')
     and a set of parameters as a Python dictionary, return a Python dictionary containing the result.
  """
  if params:
    data = urllib.urlencode(params)  # Turn the dictionary into a string with encoded 'name=value' pairs
  else:
    data = ''
  # Get the data
  try:
    result = json.load(urllib2.urlopen(BASEURL + servicetype + '/' + service + '?' + data))
  except urllib2.HTTPError as error:
    print "HTTP error from server: code=%d, response:\n %s" % (error.code, error.read())
    return
  except urllib2.URLError as error:
    print "URL or network error: %s" % error.reason
    return
  # Return the result dictionary
  return result


def get_observation(obsid=None):
  """Get an observation structure from the metadata web service, given an obsid.
  """
  if not obsid:
    return None

  obs = getmeta(servicetype='metadata', service='obs', params={'obs_id':obsid})
  return obs