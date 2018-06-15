import asyncio
import aiohttp
import numpy as np
from gcloud.aio.auth import Token
from .scopes import MLENGINE_SCOPE

BASE_URL = 'https://ml.googleapis.com/v1'
DEFAULT_HEADERS = {
  'Accept': 'application/json',
  'Accept-Encoding': 'gzip, deflate'
}

class MachineLearningClient(object):
  """Interface with Google Machine Learning API.

  We create a Async Context to provide a higher level interface to
  Google Cloud API using `aiohttp` and `gcloud.aio` for token generation.

  """

  def __init__(self, project: str, service_file: str, token=None, session=None):
    """Initialize a new Instance.

    You should interact inside a async context for interaction with their
    APIs.

    :param str project:
      google project name

    :param str service_file:
      path to service account json. you should give permissions to this service
      account for Machine Learning API (Owner)

    :param gcloud.aio.auth.Token token:
      (optional) a pre-initialized Google Cloud token

    :param aiohttp.ClientSession session:
      (optional) a pre-initialized aiohttp client session

    """
    self.project = project
    self.service_file = service_file
    self.session = session or aiohttp.ClientSession()
    self.token = token or Token(self.project, self.service_file,
      session=self.session, scopes=[MLENGINE_SCOPE])

  async def __aenter__(self):
    """Create a Async Context.

    Example of usage:

    ```python
      async with MachineLearningClient(...) as client:
        resp = await client.prediction(...)
    ```

    """
    return self

  async def __aexit__(self, *args, **kwargs):
    """Close the aiohttp.ClientSession"""
    await self.session.close()

  async def predict(self, model_name, instances, version=None):
    """Create a Tensorflow Prediction.

    This method will call the Google Cloud Machine Learning API to make a new
    prediction based on an existing model, using the online-prediction format.

    If you want to use the `batch-prediction`, you should instantiate a new
    method based on this, and add the Google Cloud Storage scopes.

    :param str model_name:
      Name of a preexisting model in Google Cloud

    :param (dict|list|numpy.ndarray) instances:
      Values to be used inside prediction

    :param str version:
      (optional) you can specify the version of model, otherwise will use
      the default one.

    """
    name = 'projects/{0}/models/{1}'.format(self.project, model_name)
    token = await self.token.get()
    headers = {**DEFAULT_HEADERS, **{
      'Authorization': 'Bearer {}'.format(token)
    }}

    if version is not None:
      name += '/versions/{0}'.format(version)

    name += ':predict'

    url = BASE_URL + '/' + name

    if type(instances) is 'dict' and 'instances' in instances:
      data_to_post = instances
    elif type(instances) is list:
      data_to_post = {'instances': instances}
    elif type(instances) is np.ndarray:
      data_to_post = {'instances': instances.tolist()}
    else:
      raise ValueError()

    return await self.session.post(url, json=data_to_post, headers=headers)