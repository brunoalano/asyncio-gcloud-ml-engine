<h1 align="center">asyncio-gcloud-ml-engine</h1>

<div align="center">
  <strong>Asyncio-based <code>Google Cloud Machine Learning Engine</code> Prediction & Training</strong>
</div>

<br />

<div align="center">
  <!-- Build Status -->
  <a href="https://travis-ci.org/brunoalano/asyncio-gcloud-ml-engine">
    <img src="https://img.shields.io/travis/brunoalano/asyncio-gcloud-ml-engine/master.svg?style=flat-square"
      alt="Build Status" />
  </a>
  <!-- Test Coverage -->
  <a href="https://codecov.io/github/brunoalano/asyncio-gcloud-ml-engine">
    <img src="https://img.shields.io/codecov/c/github/brunoalano/asyncio-gcloud-ml-engine/master.svg?style=flat-square"
      alt="Test Coverage" />
</div>

## Features
- __asyncio:__ better use of your cpu idle time
- __pep8 compliant:__ following best code standards
- __tests:__ full tested to keep up-to-date on socketcluster
- __high-performance prediction:__ we try to use `aio-grpc` when possible

## Example
```python
from asyncio_ml_engine import MachineLearningClient
project = 'my-gcloud-project'
service_account = './myserviceaccount.json'

async def myfunc():
  async with MachineLearningClient(project, service_account) as client:
    prediction = await client.predict('mymodel', X)
```
You can find more examples in the `examples/` subdirectory.


## Installation
```sh
$ pip install asyncio-gcloud-ml-engine
```

## License
[MIT](https://tldrlegal.com/license/mit-license)
