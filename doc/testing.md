## Continuous integration and testing for OpenQuake

The full suite of tests for the OpenQuake Engine can be run using `pytest` in the [tests](../tests/) package:

```shell
pytest tests/
```

Python packages can also be specified to run only a subset of tests. Please also refer to `pytest` 
[documentation](https://docs.pytest.org/en/latest/contents.html) for further information and command options.

```shell
pytest tests/test_general.py

pytest tests/test_PLoM.py
```
